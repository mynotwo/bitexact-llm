# perf/b128 (perf/attn + perf/elt2 merged): verification and throughput

## Conclusion

1. **Bit-identity: PASS.** Full plugin suite (`tests/vllm/run_tests.sh`, 14 runs) for perf/b128 and for 29e5cb5 on the
   same GPU in one session. Every run of both trees: token_seqs_diff 0, missing 0, logit_hash_diff 0,
   inconsistent_repeats 0; compare_exit 0 and compare_long_exit 0. All 14 `repo_X.hash` files are byte-identical
   (`cmp`) to this session's 29e5cb5 files and to the 29e5cb5 files of the perf/attn verification session
   (`b128_20261031/attn/verify/out/suite`).
2. **GPU pytest: 21 passed, 0 skipped**, with `BITEXACT_VLLM_MODEL` set so `test_overflow_check.py` runs
   (`tests/test_gpu_kernels.py`, `test_vllm_kernels.py`, `test_int8_ops.py`, `tests/vllm/test_decode_paths.py`,
   `test_overflow_check.py`). CPU tests on the Mac: 62 passed.
3. **Throughput T=0** (median of 3 rounds, trees alternating, same GPU and session): B=128 **19,712 → 23,978 tok/s
   (1.22x)**, B=32 8,226 → 9,570 (1.16x), B=1 371 → 388 (1.04x). T=1: B=128 18,300 → 19,386 (1.06x), B=32 8,049 →
   8,916 (1.11x), B=1 372 → 379 (1.02x).
4. **Long context S=16,384, decode step (graph replay):** B=8 **11.82 → 7.72 ms (1.53x)**, B=1 3.16 → 2.87 ms (1.10x).
   Prefill of a 512-token chunk unchanged (92–94 ms).
5. Against vLLM bf16 in the same session: at B=128 the plugin reaches 0.72x `VLLM_BATCH_INVARIANT=1` + `TRITON_ATTN`
   (33,387 tok/s) and 0.71x stock bf16 (33,771). At B=32 and B=1 it is 1.27x and 1.81x the batch-invariant baseline.
   The pre-combo plugin (48a60f7) measured in the same rounds: 9,341 tok/s at B=128, so the plugin is now 2.57x that.

## What was merged

- perf/attn (5601138): limb-stacked decode attention kernels in `tri_paged.py`; verification in
  `experiments/perf/b128_attn/RESULTS.md`.
- perf/elt2 (ded5391): Triton kernels for the MLP's SiLU/requant (`tri_mlp.py`) and the logits float copy + greedy
  argmax (`tri_logits.py`); verification in `experiments/perf/b128_elt/RESULTS.md`. The untracked `fused_mm.py`,
  `bench_mm.py`, `test_fused_mm.py` in that worktree were not merged.
- Both merges were clean (no conflicts). The two branches share only `tests/test_vllm_kernels.py`, where git combined
  the attention test change with the new logits and MLP tests.

## Setup

- Server 141, H800. Runner `experiments/perf/b128/run.sh`, trees exported with `git archive` (perf/b128 → `repo`,
  29e5cb5 → `base`, 48a60f7 → `pre`), outputs in `/data41/audreymao/b128_20261031/merge/out/`.
- Check lane, GPU 1, 03:50–04:14 server clock: `STAGES="pytest suite"`, then `pytest2.sh` (pytest with the model
  variable set; the runner's first pytest pass ran without it and skipped the overflow test, 20 passed, 1 skipped).
- Perf lane, GPU 7, 03:51–04:32: `STAGES="bench longctx"`. One run at a time; each run waits for 0 MiB and no compute
  process on its GPU.
- Bench: `tests/vllm/bench.py` (128 input / 128 output tokens per request, `max_num_seqs` = batch, second of two
  identical generate calls timed). vLLM rows: `experiments/perf/det_compare/bench_attn.py` on the bf16 weights, stock
  and with `VLLM_BATCH_INVARIANT=1 --attn TRITON_ATTN`. Per round and batch: T=0 base/repo/pre (order rotates per
  round), bf16 stock, BI, then T=1 base/repo (order alternates).

## Bench (output tok/s; runs r1 r2 r3, median)

| cell | B=1 | B=32 | B=128 |
|---|---|---|---|
| vLLM bf16 stock | 603 609 605 → **605** | 15,991 15,808 16,038 → **15,991** | 33,771 33,152 33,901 → **33,771** |
| vLLM bf16 BI + TRITON_ATTN | 214 215 214 → **214** | 7,510 7,519 7,515 → **7,515** | 34,559 33,387 32,797 → **33,387** |
| plugin 48a60f7 (pre-combo), T=0 | 235 237 225 → **235** | 4,414 4,410 4,460 → **4,414** | 9,341 9,327 9,373 → **9,341** |
| plugin 29e5cb5, T=0 | 367 371 376 → **371** | 8,214 8,231 8,226 → **8,226** | 19,712 19,587 19,738 → **19,712** |
| plugin perf/b128, T=0 | 382 388 388 → **388** | 9,505 9,672 9,570 → **9,570** | 23,779 23,978 24,038 → **23,978** |
| plugin 29e5cb5, T=1 | 370 374 372 → **372** | 8,049 8,028 8,095 → **8,049** | 18,348 18,300 18,211 → **18,300** |
| plugin perf/b128, T=1 | 378 379 381 → **379** | 8,916 8,907 9,024 → **8,916** | 19,418 19,386 19,267 → **19,386** |

The T=1 gain at B=128 (1.06x) is smaller than at T=0 (1.22x): the greedy argmax kernel from perf/elt2 serves only T=0
rows, and sampled rows still go through the integer Gumbel path.

## Long context (S = 16,384, `tests/vllm/longctx_step.py`, CUDA event median)

| tree | B | decode ms/step (graph) | decode ms/step (eager) | prefill 512-token chunk |
|---|---|---|---|---|
| 29e5cb5 | 1 | 3.16 | 9.30 | 91.7 ms |
| perf/b128 | 1 | 2.87 | 9.53 | 93.2 ms |
| 29e5cb5 | 8 | 11.82 | 12.96 | 92.3 ms |
| perf/b128 | 8 | 7.72 | 10.92 | 93.6 ms |

## Files

- `out/SUMMARY_pytest_suite.txt`, `out/SUMMARY_bench_longctx.txt`: runner summaries (per-run results, medians, cmp).
- `out/suite/{base,repo}_*.{hash,json,log}`, `out/pytest/`, `out/pytest2/`, `out/bench/*_r{1,2,3}.log`,
  `out/longctx/*.log`.
