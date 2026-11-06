# p15 route1b: 1B exact linears on the stacked GEMM path

This change routes selected 1B exact-mode linear shapes at 17 to 128 rows to the existing stacked path (`_stacked_linear`), either with `torch._int_mm` or with the wgmma configuration `(128,128,128,4,8,5)`. Only the routing block in `FusedLin.__call__` changed. Kernels, `_epilogue`, `_stack` and the TABLE are unchanged. Baseline is `eb1b09f`; the candidate is branch `perf15/route1b` with the routing commit. Host: 141 (H800), GPUs 0 and 1 only.

## Routing

Each row is an M range, extended to the midpoint between sampled M values. Rows with no entry keep the fused `_lin_kernel` path.

| shape (name, N, K) | torch._int_mm | wgmma |
|---|---|---|
| qkv (3072, 2048) | 17-20, 41-104 | 105-128 |
| o (2048, 2048) | 17-20, 89-104 | 73-88 |
| gate_up (16384, 2048) | - | 73-120 |
| down (2048, 8192) | 17-40 | 41-128 |
| lm head (128256, 2048) | 17-20 | 21-40, 73-88, 105-120 |

Caveat: the ranges between sampled M values are midpoint extensions, not measured optima. The sampled M values are 17, 24, 32, 48, 64, 80, 96, 112 and 128. The held-out values 20, 40, 72, 100 and 120 were used to check the extension. M 1-16 and M 129+ are unchanged.

## Sweep (sampled M, `sweep.py`, 3 layers, 3 alternating CUDA-graph pairs)

Ratio = fused time / variant time, so above 1 means the variant is faster. The routing rule requires at least 1.03 in every pair. Every output and flag matched (90 of 90 cells, `torch.equal`).

| shape | M | fused cfg | torch._int_mm ratio [min pair] | wgmma ratio [min pair] |
|---|---:|---|---:|---:|
| wqkv (3072,2048) | 17 | (16, 64, 128, 4, 3) | 1.185 [1.185] | 1.015 [1.014] |
| wqkv (3072,2048) | 24 | (16, 64, 128, 4, 3) | 0.911 [0.910] | 1.002 [1.002] |
| wqkv (3072,2048) | 32 | (16, 64, 128, 4, 3) | 0.910 [0.908] | 1.004 [1.003] |
| wqkv (3072,2048) | 48 | (64, 32, 128, 4, 3) | 1.235 [1.233] | 0.938 [0.937] |
| wqkv (3072,2048) | 64 | (64, 32, 128, 4, 3) | 1.199 [1.195] | 0.932 [0.931] |
| wqkv (3072,2048) | 80 | (64, 32, 128, 4, 3) | 1.302 [1.301] | 1.232 [1.231] |
| wqkv (3072,2048) | 96 | (64, 32, 128, 4, 3) | 1.244 [1.244] | 1.154 [1.152] |
| wqkv (3072,2048) | 112 | (64, 32, 128, 4, 3) | 1.056 [1.055] | 1.160 [1.160] |
| wqkv (3072,2048) | 128 | (64, 32, 128, 4, 3) | 1.032 [1.031] | 1.141 [1.141] |
| wo (2048,2048) | 17 | (16, 32, 128, 4, 3) | 1.082 [1.079] | 0.764 [0.761] |
| wo (2048,2048) | 24 | (16, 32, 128, 4, 3) | 0.879 [0.878] | 0.747 [0.746] |
| wo (2048,2048) | 32 | (16, 32, 128, 4, 3) | 0.882 [0.879] | 0.749 [0.747] |
| wo (2048,2048) | 48 | (16, 64, 128, 4, 3) | 0.926 [0.925] | 0.993 [0.991] |
| wo (2048,2048) | 64 | (16, 64, 128, 4, 3) | 0.967 [0.966] | 0.988 [0.988] |
| wo (2048,2048) | 80 | (64, 32, 128, 4, 4) | 1.209 [1.206] | 1.245 [1.243] |
| wo (2048,2048) | 96 | (64, 32, 128, 4, 4) | 1.201 [1.201] | 0.893 [0.890] |
| wo (2048,2048) | 112 | (64, 32, 128, 4, 4) | 0.964 [0.964] | 0.905 [0.903] |
| wo (2048,2048) | 128 | (64, 32, 128, 4, 4) | 0.939 [0.939] | 0.899 [0.897] |
| wgu (16384,2048) | 17 | (16, 128, 128, 4, 3) | 0.917 [0.916] | 0.870 [0.866] |
| wgu (16384,2048) | 24 | (16, 128, 128, 4, 3) | 0.809 [0.809] | 0.839 [0.837] |
| wgu (16384,2048) | 32 | (16, 128, 128, 4, 3) | 0.798 [0.795] | 0.822 [0.821] |
| wgu (16384,2048) | 48 | (64, 64, 128, 4, 2) | 0.939 [0.938] | 0.820 [0.819] |
| wgu (16384,2048) | 64 | (64, 64, 128, 4, 2) | 0.855 [0.854] | 0.856 [0.855] |
| wgu (16384,2048) | 80 | (64, 64, 64, 4, 3) | 1.241 [1.238] | 1.402 [1.402] |
| wgu (16384,2048) | 96 | (64, 64, 64, 4, 3) | 1.041 [1.040] | 1.060 [1.059] |
| wgu (16384,2048) | 112 | (64, 64, 64, 4, 3) | 0.887 [0.887] | 1.055 [1.054] |
| wgu (16384,2048) | 128 | (64, 64, 64, 4, 3) | 0.854 [0.852] | 1.001 [1.000] |
| wd (2048,8192) | 17 | (16, 32, 256, 4, 3) | 1.923 [1.919] | 1.414 [1.414] |
| wd (2048,8192) | 24 | (16, 32, 256, 4, 3) | 1.499 [1.498] | 1.395 [1.394] |
| wd (2048,8192) | 32 | (16, 32, 256, 4, 3) | 1.133 [1.132] | 1.043 [1.043] |
| wd (2048,8192) | 48 | (16, 64, 256, 4, 3) | 1.356 [1.356] | 1.441 [1.438] |
| wd (2048,8192) | 64 | (16, 64, 256, 4, 3) | 1.350 [1.349] | 1.434 [1.434] |
| wd (2048,8192) | 80 | (64, 32, 128, 4, 4) | 1.754 [1.752] | 1.823 [1.821] |
| wd (2048,8192) | 96 | (64, 32, 128, 4, 4) | 1.023 [1.022] | 1.120 [1.120] |
| wd (2048,8192) | 112 | (64, 32, 128, 4, 4) | 1.013 [1.011] | 1.141 [1.139] |
| wd (2048,8192) | 128 | (64, 32, 128, 4, 4) | 0.993 [0.993] | 1.123 [1.122] |
| lm (128256,2048) | 17 | (32, 64, 128, 4, 3) | 1.213 [1.212] | 1.171 [1.170] |
| lm (128256,2048) | 24 | (32, 64, 128, 4, 3) | 1.035 [1.035] | 1.095 [1.094] |
| lm (128256,2048) | 32 | (32, 64, 128, 4, 3) | 1.004 [1.004] | 1.058 [1.057] |
| lm (128256,2048) | 48 | (64, 64, 64, 4, 4) | 0.927 [0.927] | 0.746 [0.744] |
| lm (128256,2048) | 64 | (64, 64, 64, 4, 4) | 0.854 [0.853] | 0.872 [0.871] |
| lm (128256,2048) | 80 | (64, 64, 64, 4, 4) | 1.233 [1.209] | 1.421 [1.420] |
| lm (128256,2048) | 96 | (64, 64, 64, 4, 4) | 1.036 [1.034] | 1.005 [1.002] |
| lm (128256,2048) | 112 | (64, 64, 64, 4, 4) | 0.879 [0.879] | 1.065 [1.064] |
| lm (128256,2048) | 128 | (64, 64, 64, 4, 4) | 0.848 [0.847] | 1.011 [1.010] |

## Held-out check (`holdout.py`, M = 20, 40, 72, 100, 120, routed `__call__` vs fused TABLE path)

Ratio = fused / routed. Every output and flag matched. The routed choice is at least 0.97 in every cell. The gate_up cell at M=20 (0.991) has no route, so both sides run the same fused path and the value is noise.

| shape | M | equal | ratio [min pair] |
|---|---:|---|---:|
| wqkv (3072,2048) | 20 | True | 1.245 [1.239] |
| wqkv (3072,2048) | 40 | True | 1.001 [1.001] |
| wqkv (3072,2048) | 72 | True | 1.421 [1.414] |
| wqkv (3072,2048) | 100 | True | 1.334 [1.329] |
| wqkv (3072,2048) | 120 | True | 1.225 [1.224] |
| wo (2048,2048) | 20 | True | 1.122 [1.120] |
| wo (2048,2048) | 40 | True | 1.005 [1.004] |
| wo (2048,2048) | 72 | True | 0.999 [0.999] |
| wo (2048,2048) | 100 | True | 1.322 [1.320] |
| wo (2048,2048) | 120 | True | 1.000 [0.999] |
| wgu (16384,2048) | 20 | True | 1.002 [0.991] |
| wgu (16384,2048) | 40 | True | 0.998 [0.997] |
| wgu (16384,2048) | 72 | True | 0.999 [0.998] |
| wgu (16384,2048) | 100 | True | 1.170 [1.168] |
| wgu (16384,2048) | 120 | True | 1.071 [1.070] |
| wd (2048,8192) | 20 | True | 1.836 [1.834] |
| wd (2048,8192) | 40 | True | 1.461 [1.460] |
| wd (2048,8192) | 72 | True | 1.977 [1.972] |
| wd (2048,8192) | 100 | True | 1.273 [1.273] |
| wd (2048,8192) | 120 | True | 1.252 [1.248] |
| lm (128256,2048) | 20 | True | 1.201 [1.200] |
| lm (128256,2048) | 40 | True | 1.213 [1.211] |
| lm (128256,2048) | 72 | True | 0.999 [0.999] |
| lm (128256,2048) | 100 | True | 0.999 [0.999] |
| lm (128256,2048) | 120 | True | 1.090 [1.089] |

## Correctness (141, GPUs 0/1, `run.sh`)

| gate | base eb1b09f | repo | counts |
|---|---|---|---|
| pytest (GPU, all BITEXACT_* set) | 147 passed, 8 skipped | 147 passed, 8 skipped | matches p14 |
| 1B exact suite, 14 runs | short compare exit 0; long compare exit 0 (rerun) | short exit 0; long exit 0 | 14 of 14 hash files identical, all non-empty |
| tokens exact mixed / greedy / logprobs | IDENTICAL / IDENTICAL / IDENTICAL (logprobs rerun) | | 3 of 3 identical |

Base long_alone and base logprobs first ran under a SIGTERM (rc 143) with no error in their logs. They were rerun with identical settings and both results matched the repo. The cause is unknown; see anomalies.

## Throughput (`tests/vllm/bench.py`, exact 1B, 128 in / 128 out, T=0, 5 reps)

Each rep starts a fresh engine. Base and repo alternate on GPU 0, with order reversed on odd reps. Tokens/s, median [min, max].

| B | base | repo | change |
|---|---:|---:|---:|
| 1 | 254.9 [254.2, 259.5] | 254.5 [254.2, 255.4] | -0.2% |
| 32 | 6,950.5 [6,892.3, 7,004.4] | 7,131.8 [7,126.5, 7,244.9] | +2.6% |
| 128 | 14,207.9 [14,155.7, 14,320.5] | 14,452.6 [14,404.0, 14,552.7] | +1.7% |

Raw values are in `bench_results.jsonl`.

## Decode profile (B=128, T=0, one trace per tree, ms per step)

From `prof_buckets.md`. The gemm_int8 bucket is the limb-multiply kernels, and the fused kernel name is `_lin_kernel`.

| tree | kernel_sum | gemm_int8 | gemm_bf16 (`_wgmma`) | other (incl. `_epilogue`) |
|---|---:|---:|---:|---:|
| base | 7.957 | 4.729 | 0.000 | 0.196 |
| repo | 7.727 | 2.649 | 1.487 | 0.561 |

Linear buckets: base gemm_int8 4.729 ms; repo gemm_int8 + gemm_bf16 + `_epilogue` = 2.649 + 1.487 + 0.260 = 4.396 ms. Decode kernel sum is 7.957 to 7.727 ms, a reduction of 0.230 ms per step. Prefill is unchanged in practice (667.476 vs 667.415 ms).

## Anomalies

- Two base jobs (`base_exact_logprobs`, `base_exact_long_alone`) were killed with SIGTERM at 03:03 with no error in their logs. The repo driver contains no kill calls. Cause unknown. Both were rerun and matched.
- The first corr-stage compare reported `cmp DIFF exact_long_alone` and `tokens exact logprobs DIFF`. Both were artifacts of the missing base files, not output differences; the reruns matched.
- The base suite printed a `plan_stats.py` FileNotFoundError from the stats helper. It did not affect the compare verdicts.
- Large sweep ratios (wd at M=17 at 1.92x, held-out wd at 1.46-1.97x) are measured, but the end-to-end bench gain at B=1/32/128 is much smaller. These are kernel-level ratios on a single shape, not a model-level speedup.
- At B=1 the change is flat (-0.2%, within the min-max overlap). The routing only affects M 17 and above.
- Earlier, two driver processes appeared to race on the same out files. The process tree shows one driver. Its background subshells carry the same command line.

## Reproduction

`run.sh sync`, then `STAGES="holdout corr"`, then `STAGES="rerun bench prof"`. Remote root `/data41/audreymao/perf15_20261106/route1b`. The driver holds a flock on `out/driver.lock`.
