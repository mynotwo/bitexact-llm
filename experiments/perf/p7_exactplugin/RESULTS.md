# p7_exactplugin: exact mode (original bf16 weights) in the vLLM plugin

## Conclusion

- Exact mode now runs inside the vLLM plugin (Llama-3.2-1B and Llama-3.1-8B), with vLLM's scheduler, paged KV cache, prefix cache, CUDA graphs and the integer sampler. The mode is chosen by the constants file (exact constants contain `lm.limbs`).
- **Bit identity vs the standalone exact engine: PASS.** 1B: the 14-run invariance suite gave 0 token differences and 0 int64-logit-hash differences on the final commit (details below). 8B: 192 of 192 (prompt, step) rows equal.
- Speed (1B, one H800, 128 in / 128 out, T=0, tok/s median [min-max], 5 reps interleaved): exact 212 / 5,300 / 10,628 at batch 1 / 32 / 128. vLLM bf16 stock is 612 / 16,001 / 33,651, vLLM batch-invariant (BI, TRITON_ATTN) 213 / 7,542 / 34,495. So customers' float model served bit-exactly runs at 0.35x (batch 1), 0.33x (32), 0.32x (128) of stock vLLM bf16, equal to BI at batch 1 and 0.70x / 0.31x of BI at 32 / 128. Plugin int8 (a different, quantized model) is 549 / 13,830 / 30,481.
- The first working version (existing standalone exact ops, 9 separate GEMM passes) was 2.1x slower at batch 128 (see ablation).

## What changed

- `vllm_plugin/exact_run.py`: `ExactEngine`, the token-row forward. Each token is one row (own position, slot, block-table row); the linears/norm/RoPE/SiLU/attention-output stages are `exact/segs.py` stages unchanged, so the integers are the standalone engine's. Device tensors only, so it runs under vLLM's CUDA-graph capture.
- `vllm_plugin/tri_exact_attn.py`: paged exact attention reading the same limb-plane cache as the int8 path. Same two-pass structure as `tri_paged` (max, then weights from the exact table `2^-f` with linear interpolation), exact-mode bit widths (K at 17 bits, V at 19, weights at 22, per-row q exponent).
- `vllm_plugin/exact_gemm.py`: one Triton kernel per linear: 3 activation limbs x 3 weight limbs = 9 int8 dots into 5 int32 accumulators, int64 join, sparse residual, `combine` + `shift_round_hl` rounding and the lost-bit flag, all in the epilogue. Replaces `wmm_triton` (split-K int64 partials in memory) + `lin_y`. Config table swept per (N, K, rows) (`sweep_gemm.py`); integers do not depend on it.
- `model.py` (mode switch, `_forward_exact`, exact RoPE table extension at 30 bits, logits path), `sampler.py` (logit scale `FR` from the model: exact is 24, int8 is 16), `segs.py` (variants of three stages taking the linear's output), `tests/vllm/run_ref.py` (reference on exact constants), `tests/vllm/bench.py` (`--attn`).
- Not reused from the int8 plugin: `tri_fuse` / `tri_mlp` / `tri_sample` kernels (their bit widths and tables are int8-mode specific; none was rewritten for exact). Elementwise stages stay as torch.compile regions.

## Bit identity (final commit)

| Check | Result |
|---|---|
| `tests/vllm/run_tests.sh`, exact constants, 10 short-target runs (alone pc on/off, busy mixed batches pc on/off, budget 64, 160-block cache with 5 preemptions, full eager, vLLM graphs busy and alone) | 0 token-seq diffs, 0 logit-hash diffs, 576 positions each |
| 3 long-target runs (5k/9k/14k tokens: alone, busy with prefix cache, busy eager) | 0 / 0, 48 positions each. `long_alone` and `long_busy_eager` from rerun `s7`; `long_busy` from the suite and two reruns (`s8`) |
| Unit tests: fused linear vs `Lin` + `lin_y` (5 layer kinds, rows 1 to 1000, three activation ranges, value and flag), exact attention vs `softmax_w` + exact products (head_dim 64/128, nsplit None/1/5/64) | pass (`torch.equal`) |
| `pytest -q tests/` with exact and int8 constants | 120 passed, 15 skipped |
| 8B exact in plugin vs standalone 8B exact engine (4 prompts, 48 steps, vLLM graphs; also batch 1 vs 32 vs 128 self-consistency over 6,144 rows) | `BATCH_INV PASS`, 192/192 rows equal in token and logit hash |

Notes: the suite ran on shared GPUs; four runs failed on vLLM's memory-profiling check because other processes changed free memory, and were rerun (logs in `logs/`). One load of the long-busy run died on `_extend_rope`'s assertion "RoPE rows differ" (fails closed, not repeated in 4 later loads); cause unknown.

## Where time goes (decode, profiler kernel time, ms/step, 1B)

| Batch | wall | kernel sum | linears (fused kernel) | elementwise (torch.compile) | attention | sampling+lm head | kernels/step |
|---|---|---|---|---|---|---|---|
| 1 | 4.49 | 4.24 | 1.97 | 1.71 | 0.19 | 0.34 | 565 |
| 32 | 5.81 | 5.46 | 2.37 | 2.18 | 0.28 | 0.59 | 633 |
| 128 | 10.66 | 10.52 | 4.92 | 3.55 | 0.74 | 1.26 | 633 |

Before the fused linear (batch 32): 7.17 ms wall, 6.84 kernel. Prefill, 128 prompts x 128 tokens: 1.23 s kernel time with the old linear for 16k rows, 0.96 s with the fused kernel (profile `ex3`).

Linear microbench (CUDA-graph timing, old = `wmm_triton` + compiled `lin_y`, layer 0): wqkv 23.5 -> 14.3 us at 1 row, 78.6 -> 53.1 us at 128; wo 209 -> 27.6 us at 128; wgu 431 -> 62.7 us at 32 rows, 443 -> 148 us at 128; wd 205 -> 93 us at 128; lm head 9.3 ms -> 1.0 ms at 128. Exception: wd at 1 row 30.4 -> 49.3 us (slower). At 512 / 4096 rows: 1.2x to 3.4x faster.

## Throughput tables

1B, `tests/vllm/bench.py` (`bench_all.sh`), one H800, same session per table, interleaved, T=0 (`logs/bench2`; n shows valid runs; runs where another process shared the GPU at the end are flagged in `progress.log`):

| config | batch 1 | batch 32 | batch 128 |
|---|---|---|---|
| plugin exact (final) | 212 [211-213] n=4 | 5,300 [5,258-5,492] n=5 | 10,628 [10,601-10,808] n=5 |
| plugin int8 | 549 [536-585] n=5 | 13,830 [13,724-13,976] n=4 | 30,481 [29,355-30,726] n=5 |
| vLLM bf16 stock | 612 [603-631] n=5 | 16,001 [11,248-16,179] n=5 | 33,651 [20,525-34,137] n=5 |
| vLLM BI + TRITON_ATTN | 213 [212-214] n=4 | 7,542 [7,475-7,607] n=5 | 34,495 [33,978-34,707] n=5 |

The low minima for bf16 (11,248; 20,525) are runs a foreign process shared the GPU with (the first repetition on GPU 2 is flagged); medians are unaffected. An earlier set on the build that used the fused linear only up to 256 rows (`logs/bench`, GPU 2, 5 reps, a separate session) gave exact 212 / 2,117 / 5,868: the gain at batch 32 and 128 is the prefill (4k and 16k rows) using the fused kernel.

Standalone exact engine for reference (graph + compile, earlier run on the same 141 setup): 223 / 3,321 / 5,615.

8B (Llama-3.1-8B, `bench8.sh`, 3 reps, exact vs int8 plugin, same session, no foreign process flagged): exact 58.9 [58.4-61.3] / 956 [955-982] / 1,124 [1,117-1,124]; int8 180 [179-180] / 4,798 [4,030-4,863] / 6,735 [6,728-6,789] at batch 1 / 32 / 128. 8B stock vLLM and BI not measured.

Ablation (`ablate.sh`, exact plugin, fused linear vs old path via `BITEXACT_EXACT_FUSE_M=0`): batch 128 fused 10,798 [10,624-10,822] (n=3) vs old 5,356 [5,286-5,360] (n=3): 2.0x. Batch 1: fused ~211 (the 110 run was contaminated) vs old 223 [222-224]: the fused path is about 5% slower at batch 1 (the `wd` case above). Batch 32: only n=1 fused / n=2 old valid (3,015 vs 2,928), too few to conclude; the contaminated runs were not used.

## What did not work / limits

- One-region torch.compile for norm+quant (`BITEXACT_EXACT_ONEREGION`) was slower at batch 32 (7.31 vs 7.17 ms), reverted.
- Exact stays 3x below stock bf16 at batch 32/128. Remaining time at batch 128: linears 47%, elementwise 34%, attention 7%. The linears run 9 int8 dot products per K step; the int8 plugin does 2 to 3.
- The elementwise stages (RMS, quantization, SiLU, RoPE, attention-output division with int64 long division) are still ~600 Inductor kernels per step; the int8 plugin's `tri_fuse` kernels (11 per layer) were not ported to exact bit widths.
- 8B exact is 6.5x to 3x slower than 8B int8 (batch 1 to 128); not tuned (the GEMM config table covers 1B shapes only, 8B uses the default configs).

## Not verified / not done

- Exact vs HF accuracy was not re-run (README numbers apply: the integers equal the standalone engine's).
- Long-context (> 14k) and speculative decoding with exact mode: not tested.
- 8B suite: only the batch-invariance script (4 prompts), not the full 14-run suite; 8B stock vLLM / BI throughput not measured.
- T>0 sampling is covered by the suite targets (T 0.7 / 1.0) only; the noisy-temperature fast path (`tri_sample.noisy`) is int8-only, exact uses `integer_sample`.
- Logprobs returned to vLLM are approximate (integer logits / 2^24 as float32), as in the int8 plugin.

## Files

`run_tests`/`bench` drivers: `sync.sh`, `r.sh`, `suite.sh`, `launch_free.sh`, `bench_all.sh`, `bench8.sh`, `ablate.sh`, `prof.sh`, `prof_vllm.py`, `buckets.py`, `bench_gemm.py`, `sweep_gemm.py`, `ref8.py`, `cmp8.py`, `agg.py`. Logs in `logs/`.
