# Plugin decode speed-up (exp/speed-decode, from exp/profile e5eb24c)

One H100 (host 138, one GPU per run, each started on a GPU showing 0 MiB), torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0. Nothing under `src/bitexact/int8/` or `bmm.py` was edited. Changes are in `src/bitexact/vllm_plugin/` only.

## Result

ms/step = wall of a 512-token generate (prefill included) / 512, T = 1, ignore_eos, B requests of 178-260-token GSM8K prompts, vLLM FULL_DECODE_ONLY graphs, median of 3 (`speed.py`). tok/s = generated tokens / wall.

| item | B=1 ms/step | B=1 tok/s | B=64 ms/step | B=64 tok/s | bit gate |
|---|---|---|---|---|---|
| before (e5eb24c) | 4.286 | 233.3 | 18.160 | 3,524 | reference |
| a. static-shape lm_head step (`vllm_plugin/logits.py: lm_logits`) | 4.102 | 243.8 | 9.899 | 6,465 | PASS |
| b. no host syncs in `compute_logits` and the sampler | 2.679 | 373.3 | 6.985 | 9,162 | PASS |
| total | 1.60x | | 2.60x | | |

c. No further item committed (see "Not done").

## Bit-exactness gate (every item)

Workload: first 32 GSM8K test questions, 2-shot prefix, G = 8, request seeds 1 + 8p + g, max 256 new tokens, stop at EOS (the portability `sample.py`), T = 1 (53,387 tokens) and greedy (65,536 tokens), `max_num_seqs` 1 and 64. `gate.py` compares tokens, argmax, log p (float64 from the int64 logits row) and the logits-row SHA-256 prefix with plain equality. After item a and after item b, for T = 1 and greedy: new vs before at B=1: 256/256 sequences identical, 0 differences; new vs before at B=64: same; B=1 vs B=64: same. The gate runs of item b ran with `BITEXACT_VERIFY_HOST_ROWS=1`, which asserts that the host-side row positions equal the device ones.

## Cross-GPU

- Integer engine (`bitexact_engine.py score`, H100) scoring the plugin's final rollouts (B=1 and B=64, T = 1 and greedy): 256/256 sequences, 0 log p differences, 0 logits-row hash differences, all four files.
- H800 vs H100: 141 has no environment that runs vLLM 0.32 here (no torch in the home or `/data08/audreymao` venvs; the portability work found the cu130 wheels do not start on 141's driver 535), so no new H800 run was made. Instead the final H100 plugin rollouts (B=64, T = 1) were compared with the H800 plugin rollouts recorded earlier by abc_measure (`h800_q1_B.json`, produced by the old code): 256/256 sequences identical, 53,387 tokens, 0 log p differences, 0 logits-row hash differences.

## Item a: root cause of the single-CTA reduction

`Segs.post` (`int8/segs.py`) is one `torch.compile(dynamic=None)` function used by every GEMM epilogue (q/k/v, o, gate/up, down, lm_head). It ends in the overflow flag `t.abs().amax() >= 2^38`, a reduction of the whole [M, N] output to one element. The first compile (static shapes) splits that reduction over many CTAs. The first call with another row count or width makes Dynamo recompile with symbolic sizes, and the Inductor kernel for that recompile has grid (1, 1, 1): one CTA with 512 threads loops over all M x N elements (the cached kernel has `xnumel = 1`, a runtime `r0_numel`, and no split). vLLM's warm-up and capture runs use many row counts, so by decode time lm_head runs the symbolic kernel: 8.1 ms per step at M = 64 (M x 128,256 int64 = 8.2M elements).

Evidence: `rootcause.py`, lm_head for 64 rows after six recompile orders (second shape 2 to 512 rows, different weights) takes 3.9 ms every time against 0.09-0.3 ms for the static kernel, so the slowdown does not depend on the size of the recompile's shape. The same kernel is present in the vLLM trace as grid (1,1,1), 8.1 ms, once per decode step. The exact Inductor decision that withholds the split for symbolic reduction sizes was not traced to a source line; this is the part left unverified.

Fix: `lm_logits` runs the final norm + quantization and the GEMM epilogue as plugin-owned `torch.compile(dynamic=False)` functions for row buckets 1, 2, 4, ..., 512 (extra rows repeat the last real row and are dropped). Static shapes give the split reduction (0.30 ms for 64 rows after the same sequences). The same integer ops are executed; rows are independent, the flag is a maximum, and repeated rows cannot raise it. `tests/test_vllm_logits.py` checks equality with `IntLlama.logits` for row counts 1 to 130 and that the flag still fires.

## Item b: host side

`cProfile` over 512 B=1 steps (`results/cprof_B1.log`) put the time in vLLM's step loop and in `IntSampler.apply`, whose `.tolist()` on `expanded_idx_mapping` and `pos` block until the step's kernels finish, plus `bool(self.err)` in `check()`. Changes: `DeferredCheck` copies the overflow flag to pinned memory with an event and raises one step later (the flag is sticky; a violation still aborts the run); the sampler reads row to slot and generated-token index from vLLM's host arrays (`idx_mapping_np`, `seq_lens_upper_bound_np`) when there is one logits row per request, and skips the row gather when all rows share one temperature. vLLM's own CUDA-graph capture already covers the plugin model (`_captured_decode`); nothing changed there. B=1 step time ends below the profile's standalone graphed step (2.89 ms).

## Not done

- Prefill still runs the single-CTA kernel: in a 64-step profile window the prefill step shows 16 pairs of grid (1,1,1) kernels (1.3 and 8.1 ms) from `Segs.post` / `post_add` in `paged.mlp` and `paged._out` (about 0.15 s per 64-request prefill, about 4% of a 512-token B=64 generate). A plugin-owned compile does not help directly because prefill row counts are unbounded. Suggested change in `int8/segs.py` (not made): compute the overflow flag in a separate reduction outside the fused epilogue, or compile `post` / `post_add` per call site.
- Remaining B=64 time is attention (`_acc_kernel`, `_max_kernel`, about 0.77 + 0.33 ms per step) and the int8 GEMMs; no cheap win there was found in this pass.
- No fresh H800 run (see Cross-GPU).

## Files

`speed.py` (ms/step, tok/s), `gate.py` (bit comparison), `sample.py` / `compare.py` (portability copies), `rootcause.py`, `run138.sh` / `env138.sh` / `sync.sh` (138 runner), `results/` (raw timings, `rootcause.json`).
