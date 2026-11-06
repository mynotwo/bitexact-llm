# p14_verify: combined verification of p10 to p13

This run gates the current `main` line before push. It covers two trees: `base` = `6faf8ef` (origin/main, includes the p9 four-lane merge) and `repo` = `e74549f` (local main = base + `dd01d63` p10 exact GEMM config tuning + `e74549f` stacked GEMM and integer reciprocals in `exact_gemm.py` and `exact_fuse.py`). The source diff between the two trees is `src/bitexact/vllm_plugin/exact_gemm.py`, `src/bitexact/vllm_plugin/exact_fuse.py`, `tests/test_vllm_exact_gemm.py`, plus the p10 experiment files. All harness scripts (`tests/vllm/*`, `experiments/perf/p4_merge/tokens.py`, `experiments/perf/p7_exactplugin/*`, `experiments/abc_rl/check_plug.py`) are byte-identical across the two trees.

Verdict: every correctness gate passes on both trees. No `DIFF` line appears in the driver summary, no stage returned a non-zero exit, and every count below was checked against the raw outputs on 141.

## Correctness

Environment for all GPU jobs: `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `BITEXACT_INT8_CONST`, `BITEXACT_INT8_CONST_8B`, `BITEXACT_EXACT_CONST`, `BITEXACT_MODEL` and `BITEXACT_VLLM_MODEL` all set.

### Pytest (full GPU suite, `-rs`)

| Tree | Result |
|---|---|
| base `6faf8ef` | 147 passed, 8 skipped |
| repo `e74549f` | 147 passed, 8 skipped |

The 8 skips are identical on both trees and have one reason, `exact constants only`: 5 in `tests/test_vllm_exact_fuse.py` (lines 29, 54, 81, 95, 112) and 1 in `tests/test_vllm_exact_gemm.py` (line 14). These are the int8-constant cases, which the exact-only environment does not exercise.

### 1B suites (`tests/vllm/run_tests.sh`, 14 jobs per mode)

| Mode | Tree | `compare_exit` | `compare_long_exit` | Hash comparisons base vs repo |
|---|---|---|---|---|
| exact | base / repo | 0 / 0 | 0 / 0 | 14 of 14 `cmp OK` |
| int8 | base / repo | 0 / 0 | 0 / 0 | 14 of 14 `cmp OK` |

Each short-target reference hash file holds 576 rows (12 targets x 48 positions); each long-target file holds 48 rows (3 targets x 16 positions). The 14 jobs are `alone_pc1`, `alone_pc0`, `busy_pc1_s0`, `busy_pc0_s1`, `busy_pc1_s2_b64`, `busy_pc1_s3_preempt`, `busy_eager_s4`, `busy_vllmgraph_s5`, `alone_vllmgraph`, `long_ref`, `long_alone`, `long_busy`, `long_busy_eager`, `ref`. The forced-preemption job (`busy_pc1_s3_preempt`) writes 2,535 hash rows against the 576 of a normal short run, which is consistent with vLLM recomputing preempted steps (`compare.py` accepts repeated rows only when they agree).

### 8B exact check (batch invariance and reference, 4 prompts x 48 tokens)

| Tree | Eager 0 (CUDA graphs) | Eager 1 |
|---|---|---|
| base | 192 rows matched, 0 diff | 192 rows matched, 0 diff |
| repo | 192 rows matched, 0 diff | 192 rows matched, 0 diff |

Batch sizes 1, 32 and 128 are invariant against the standalone reference (`ref8.py`).

### Production token dumps (base vs repo, `cmp` on JSON)

| Mode | mixed temperatures | greedy (T=0) | logprobs |
|---|---|---|---|
| exact | IDENTICAL | IDENTICAL | IDENTICAL |
| int8 | IDENTICAL | IDENTICAL | IDENTICAL |

### check_plug.py (abc_rl plugin check), both trees

Exit code 0 on both trees. Four updates each; every differ field (`seq_tokens_differ`, `sampler_lp_differ`, `rescore_lp_differ`, `plugin_sampler_vs_rescore_differ`, `intq_sampler_vs_rescore_differ`) is 0 in all four updates on both trees. Token counts per update: 14,470 / 13,571 / 12,283 / 13,734.

The bound-check entry is a deliberate negative test. The script multiplies `0.wqkv.m` by 2^20 and expects an `OverflowError` with the weights left unchanged. Both trees raised `post_bound 3329570070921216 >= 2^38 for 0.wqkv` and left the weights unchanged (`unchanged True`).

### spec_check.py (ngram speculative decoding, suite workload, 12 targets x 96 tokens = 1152 tokens)

`k=0` and `k=3` produce byte-identical token files on both trees, and base equals repo. Both trees' RESULT lines are present (`tokens 1152`).

## Throughput

Protocol: `tests/vllm/bench.py`, 128 input / 128 output tokens, T=0 unless marked, 5 independent engine processes per cell. Base and repo alternate on the same GPU (GPU 7, idle at start, 0 MiB), with the order reversed on odd reps. No observation is excluded; every cell has n=5. Values are output tokens/s, median [min, max].

Mode key: `exact` = original bf16-weight exact mode (product path), `int8` = quantized weights (different model), `vLLM bf16` = stock vLLM on bf16 weights with CUDA graphs, `vLLM BI` = `VLLM_BATCH_INVARIANT=1` with `--attn TRITON_ATTN`. The two vLLM rows run only on the repo tree, as in `readme_bench.sh`.

### 1B

| Mode | Tree | B=1 | B=32 | B=128 |
|---|---|---|---|---|
| exact, T=0 | base `6faf8ef` | 255.5 [254.2, 255.9] | 6,806.9 [6,730.5, 6,807.5] | 13,912.6 [13,879.9, 13,962.4] |
| exact, T=0 | repo `e74549f` | 255.6 [253.9, 255.7] | 6,958.9 [6,942.4, 7,008.8] | 14,183.2 [14,124.2, 14,256.1] |
| exact, T=1 | base `6faf8ef` | 249.5 [248.8, 250.6] | 6,375.5 [6,329.8, 6,382.2] | 12,480.9 [12,449.9, 12,512.8] |
| exact, T=1 | repo `e74549f` | 249.6 [249.0, 250.4] | 6,513.8 [6,510.3, 6,561.5] | 12,699.3 [12,683.1, 12,747.7] |
| int8, T=0 | repo `e74549f` | 551.9 [542.2, 553.0] | 14,471.9 [12,988.1, 14,664.3] | 33,251.6 [32,758.3, 33,931.7] |
| vLLM bf16 (stock, CUDA graphs), T=0 | repo tree | 608.6 [605.6, 615.6] | 15,919.7 [15,784.4, 15,929.8] | 33,387.0 [32,835.3, 33,867.9] |
| vLLM BI + TRITON_ATTN, T=0 | repo tree | 214.6 [214.4, 215.3] | 7,583.4 [7,505.2, 7,601.8] | 34,560.1 [34,143.3, 34,774.8] |

Exact repo over base (ratio of medians): T=0 B=1 +0.0%, B=32 +2.2%, B=128 +1.9%; T=1 B=1 +0.0%, B=32 +2.2%, B=128 +1.7%.

### 8B

| Mode | Tree | B=1 | B=32 | B=128 |
|---|---|---|---|---|
| exact, T=0 | base `6faf8ef` | 65.4 [65.4, 65.5] | 1,363.1 [1,362.3, 1,365.8] | 3,079.3 [3,076.1, 3,081.4] |
| exact, T=0 | repo `e74549f` | 81.4 [81.3, 81.5] | 2,000.9 [2,000.7, 2,005.1] | 3,773.2 [3,767.1, 3,780.7] |
| int8, T=0 | repo `e74549f` | 180.0 [179.8, 180.6] | 5,001.7 [4,966.2, 5,003.9] | 10,775.3 [10,737.3, 10,783.0] |
| vLLM bf16 (stock, CUDA graphs), T=0 | repo tree | 151.9 [151.5, 152.1] | 4,505.2 [4,501.3, 4,533.1] | 14,054.3 [14,049.5, 14,089.5] |
| vLLM BI + TRITON_ATTN, T=0 | repo tree | 49.7 [49.7, 51.0] | 1,750.6 [1,749.6, 1,753.4] | 10,677.4 [10,625.1, 10,679.5] |

Exact repo over base (ratio of medians): B=1 +24.5%, B=32 +46.8%, B=128 +22.5%.

Note on the 8B gain: the p10 run (`experiments/perf/p10_exact_gemm/RESULTS.md`, H800, base `6faf8ef` vs `dd01d63` only) measured +2.4% at B=128. This run's repo tree also includes `e74549f` (stacked GEMM), which moved the exact 8B GEMMs to the wgmma kernel family (see profile). This run does not have a `dd01d63`-only tree, so the split between the two commits is not measured here. The 8B exact B=128 repo median is 3,773 tok/s.

## Profile (decode, B=128, T=0)

Tool: `experiments/perf/p8_exactfuse/prof_vllm.py` (identical to the p7 copy) with `buckets.py`, decode steps 8 to 23, prefix caching off. Times are ms per decode step. These profiles run under the profiler, so their tok/s (1B base 9,352, repo 9,525; 8B base 1,985, repo 2,254) are below the bench rows above and are used only for the bucket breakdown.

| Model | Tree | Wall | Kernel sum | `gemm_int8` | `gemm_bf16` | elementwise | attention | sampling + LM head | other | memcpy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B | base | 8.359 | 8.138 | 4.917 | 0 | 1.042 | 0.741 | 1.236 | 0.196 | 0.007 |
| 1B | repo | 8.135 | 7.908 | 4.728 | 0 | 1.041 | 0.742 | 1.196 | 0.195 | 0.006 |
| 8B | base | 38.551 | 37.539 | 28.505 | 0 | 3.478 | 2.863 | 2.092 | 0.595 | 0.006 |
| 8B | repo | 30.793 | 29.864 | 2.314 | 16.190 | 2.087 | 2.880 | 1.682 | 4.706 | 0.006 |

Reading the rows:
- Linears: at 1B the bucket `gemm_int8` is 4.9 ms on both trees, so the integer GEMM is unchanged. At 8B the repo moves the exact GEMMs from `gemm_int8` (28.5 ms) to `gemm_bf16` (`_wgmma`, 16.2 ms), plus `_epilogue` (3.9 ms) and `_stack` (0.4 ms) under `other`. The 8B repo linear total is about 22.8 ms against base 28.5 ms, with the remaining 2.3 ms `gemm_int8` being the int8 limb path that stays in use.
- Elementwise: unchanged at 1B (1.04 ms both trees); at 8B 3.48 to 2.09 ms.
- Sampling and LM head: 1.24 to 1.20 ms at 1B. At 8B the LM head moves from `_lin_kernel` (1.9 ms) to `_wgmma` (1.1 ms) inside the same bucket.
- Attention: unchanged at 1B (0.74 ms); at 8B 2.86 to 2.88 ms.
- Prefill (1B, B=128, kernel sum per step): base 667.5 ms, repo 666.5 ms; elementwise 78.70 ms on both trees. The prefill rows in the report print a negative `wall-busy` value because the prefill window spans several steps, so the prefill wall column is not used here.

Full per-kernel tables are in `results/profile_buckets.md`; the four profile summaries are `results/base_1b.json`, `results/repo_1b.json`, `results/base_8b.json`, `results/repo_8b.json`.

## Deviations from the p9 template

- p9's base was `175ff6c`. Here base is `6faf8ef`, which already contains the p9 merge. So base pytest counts (147 passed, 8 skipped) equal repo counts, where p9 had 138 vs 147.
- p9 ran production token dumps in int8 mode only. This run dumps both exact and int8.
- The brief's profile directory `experiments/perf/profile/` does not exist in this tree; `experiments/perf/p8_exactfuse/prof_vllm.py` and `buckets.py` are used instead (their p7 copies are identical).
- The vLLM bf16 and BI rows run on the repo tree, as `readme_bench.sh` does. Their numbers do not depend on the plugin code.

## Anomalies and open points

- The 1B int8 B=32 cell has one low run: 12,988 tok/s, about 10% below the other four (14,378, 14,472, 14,649, 14,664). The median (14,472) is unaffected. The cause was not investigated.
- The 8B exact repo B=32 gain (+46.8%) is larger than the B=1 and B=128 gains. Not investigated; it is measured and reproducible across the 5 reps (repo 2,000.7 to 2,005.1).
- Profile tok/s is lower than bench tok/s for the same configuration (profiler overhead). Bucket values are used only as ratios and attributions.
- No failures, no retries, and no non-zero exits in the driver run. The driver ran to `ALLDONE` in one pass; the Monitor was re-armed once for watching only.
- The `ah` harness (CLAUDE.md) was not run for this commit. The write-set is only `experiments/perf/p14_verify/`, no `src/` or `tests/` file changed, and the instruction was to commit directly. Nothing was pushed or merged, and README.md was not edited.

## Reproduction

Remote root on 141: `/data41/audreymao/perf14_20261106/verify/` with `base/` (`6faf8ef`), `repo/` (`e74549f`), `out/` (raw logs, traces, `SUMMARY.txt`) and `p14/run.sh`. Environment: `source /data41/audreymao/a17_vllm/env.sh`; Python `/data41/audreymao/a17_vllm/venv/bin/python`.

Driver, from this worktree:
- `bash experiments/perf/p14_verify/run.sh sync`: `git archive` both commits to 141 and copy the driver.
- `bash experiments/perf/p14_verify/run.sh start`: nohup stages `corr bench prof` (default); GPUs 0, 1, 2, 3, 7 for correctness, GPU 7 for bench and profile. Set `STAGES` to run a subset.
- `bash experiments/perf/p14_verify/run.sh status` and `... fetch`: progress and small-file pull.

Aggregation: `python3 experiments/perf/p14_verify/agg.py DIR` over the bench RESULT lines (writes `bench_summary.json`); `python3 experiments/perf/p8_exactfuse/buckets.py --report DIR` over the profile outputs.

Small summaries kept in `experiments/perf/p14_verify/results/`: `driver_summary.txt`, `bench_summary.json`, `profile_buckets.md`, and the four profile JSONs.
