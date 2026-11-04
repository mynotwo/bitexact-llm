# p3_host: host-side work of the T>0 sampler step

Branch `perf3/host` on 07aa43c, H800 (141, GPU 7), 1B int8 model, vLLM 0.32 plugin.

## Conclusion
- Bit identity: PASS. 14/14 suite runs hash- and token-identical to base (`run_tests.sh`; `alone_vllmgraph` re-run once
  after a GPU-sharing OOM in memory profiling, identical), `pytest` 27 passed / 1 skipped (test_tri_sample_noisy_exact
  now also checks the new kernel path incl. seed/t = 0 and 2^32-1), p2_sample token dumps (128 requests, mixed
  temperatures 0/1/0.7/0.5/1.5/2, T=0 only, logprobs=1) identical to base.
- T=1 gains, T=0 unchanged (T=0 code path not touched; its differences below are run noise).
  B=128 T=1: 20.2k -> 27.3k tok/s (+35%), step wall 5.94 -> 4.08 ms with kernel time 3.80 -> 3.77 ms. Wall-minus-kernel
  went 2.14 -> 0.31 ms. B=32 T=1: 10.1k -> 10.64k (+5.5%). B=1: no change.
- (c) not done, (b) only the sampler part done (see below).

## What changed
- `tri_sample._seed_state`: numpy SeedSequence([SEED_BASE, seed, t]) -> PCG64 state, computed per row inside
  `_noise_kernel` (uint32 hashmix arithmetic, 128-bit LCG step with `_mulhi`). Replaces `gpu_pcg.seed_state`, a Python
  loop over rows that ran numpy SeedSequence/PCG64 per live row every step. Verified against it on 4,002 random
  (seed, t) pairs including 0 and 2^32-1 (0 mismatches). Seeds or t >= 2^32 fall back to the old host path.
- `IntSampler`: per-slot seed/tn/td/strict are numpy arrays; ts, seeds, temps come from fancy indexing, no per-row Python
  or list building; the kernel input is one int64 [M, 4] (seed, t, tn, td) copied through a pinned ring (no
  `pin_memory()` allocation per step).
- `_noise_kernel` writes vLLM's logits row (-inf) and `_comb_kernel` stores 0 at the token (FILL), replacing the eager
  `fill_(-inf)` + two `scatter_` launches. The `BITEXACT_HASH_LOG` / strict (logprobs etc.) paths keep the old rule.

## Evidence: wall vs kernel, ms per decode step (prof_vllm.py, torch profiler, unmodified rows are noise)
| B | T | base wall / kernel | branch wall / kernel |
|---|---|---|---|
| 1 | 0 | 1.997 / 1.780 | 1.994 / 1.778 |
| 32 | 0 | 2.409 / 2.330 | 2.579 / 2.326 |
| 128 | 0 | 3.820 / 3.474 | 3.907 / 3.469 |
| 1 | 1 | 1.983 / 1.787 | 2.001 / 1.772 |
| 32 | 1 | 3.022 / 2.444 | 2.691 / 2.433 |
| 128 | 1 | 5.938 / 3.803 | 4.081 / 3.771 |

## tok/s (bench.py 128 in / 128 out, T=1, 3 interleaved runs, all numbers)
| B | base | branch | median ratio |
|---|---|---|---|
| 1 | 447.6 / 443.8 / 447.6 | 448.4 / 449.1 / 444.8 | 1.00 |
| 32 | 10002 / 10090 / 10083 | 10639 / 10780 / 10637 | 1.055 |
| 128 | 20400 / 20258 / 20042 | 27291 / 27272 / 27315 | 1.34 |
T=0 bench (one run each, code path unchanged): B=1 447.7 vs 445.9, B=32 11,885 vs 11,149 (prof run), B=128 28,963 vs 28,515; not
reproducibly different, no T=0 change claimed.

## Not done / not verified
- (b) launches before the sampler: the LM-head (`pre_norm`, `_int_mm`, `_head_kernel`, `_comb_kernel`) run in
  `compute_logits`, outside the FULL_DECODE_ONLY graph vLLM captures (it captures only the model forward). Moving them
  in needs a runner change; not attempted. T=0 keeps ~0.35 ms wall-minus-kernel at B=128.
- (c) skipping vLLM's `_gumbel_sample_kernel`: no plugin-side sampler hook found in vLLM 0.32 without patching; not done.
- Timeline files per cell are in `/data41/audreymao/perf3_host/out/prof/{base,repo}/` on 141 (wall/kernel numbers above
  come from the same runs). T=0 bench numbers are single runs.
- GPU 7 was shared with other lanes between runs; bench3 waited for 0 MiB before each launch.
