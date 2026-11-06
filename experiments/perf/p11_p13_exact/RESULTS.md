# p11 to p13 exact GEMM and fuse rounds: results

Three rounds on the 8B exact-mode model produced commit `e74549f` (stacked GEMM, exact SiLU reuse, integer reciprocals). Exact mode means the customer's original bf16 weights on the bitexact exact path. No int8-mode number appears here. All throughput figures are 8B. No 1B end-to-end throughput was measured in these rounds.

The p11 base is `dd01d63`, the p10 candidate. Round p12 uses the p11 guarded tree as its base. Round p13 uses the p12 tree as its base.

## Verdict

- Integer checks passed in every round, with the counts listed in the correctness section. The checks compare tokens, logits hashes, or integer kernel outputs against an integer reference that runs without vLLM, or against the predecessor tree.
- 8B exact throughput, medians of five runs, went up in all three rounds at batch sizes 1, 32 and 128.
- The final p13 tree is not byte-identical to `e74549f`. Its `exact_gemm.py` and `exact_fuse.py` are identical to the commit's code after removing docstrings and comments. The byte-level differences are documentation text, one comment line, and line wrapping. See the section "Final tree compared with e74549f".
- Several first-pass kernel timing cells fell below the 0.97 ratio rule. Details are in "Anomalies and gaps".

## Result summary

| Round | Base to repo | Change (file) | Correctness | 8B exact tok/s change, B=1 / 32 / 128 | 1B throughput |
|---|---|---|---|---|---|
| p11, first variant | dd01d63 to fa71 (`6daff816`) | stacked int32 limb GEMM for selected 8B shapes (`exact_gemm.py`) | integer checks PASS; kernel-timing cells below 0.97: 6 (correct run), 3 (held-out run) | +2.6% / +9.4% / +5.5% | not measured |
| p11, guarded variant | dd01d63 to guardrepo (`8a9a3a54`) | same stacked path with narrower M gates (`exact_gemm.py`) | integer checks PASS; kernel-timing cells below 0.97: 7 (correct run), 0 (held-out run) | +1.6% / +15.1% / +14.7% | not measured |
| p12 | guardrepo to p12 repo (`a3ec99fa`, `exact_fuse.py` `2e38f75a`) | reuse exact SiLU first-pass products for 8B batches of at most 128 rows (`exact_fuse.py`) | integer checks PASS (counts below) | +7.3% / +4.0% / +2.4% | not measured |
| p13 | p12 repo to p13 repo (`0830aed0`, `exact_fuse.py` `07ffac67`) | integer reciprocal division for norm (D=4096, M<=128) and attention output (4096 = 32 x 128, HD=128, N<=128) (`exact_fuse.py`) | integer checks PASS (counts below) | +5.5% / +5.3% / +2.4% | not measured |

## Trees and source hashes

Source hash method, identical to `source()` in the runner scripts: SHA-256 over the sorted list of `src/**/*.py` files, hashing each relative path, a NUL byte, then the file bytes. Files named `._*` are skipped. The values below were recomputed from `git archive` output for the commits and from copies of the 141 trees. They match the `source_sha256` values in the runners' `result.json` files.

| Tree | Location on host 141 | src hash (first 16 hex) | `exact_gemm.py` (first 16 hex) | `exact_fuse.py` (first 16 hex) |
|---|---|---|---|---|
| `dd01d63` (p11 base, p10 candidate) | `/data41/audreymao/perf10_20261105/exact_gemm/repo` | `cbb6cda39d098...` | `91bda728bda6240c` | `703017534dde7acd` |
| p11 repo (first variant) | `/data41/audreymao/perf11_20261105/gemm/repo` | `fa71d7d407f91320` | `6daff8168c7e2a7c` | `703017534dde7acd` |
| p11 guardrepo (guarded variant) | `/data41/audreymao/perf11_20261105/gemm/guardrepo` | `5cf8d781786220c4` | `8a9a3a5435bfd0be` | `703017534dde7acd` |
| p12 repo | `/data41/audreymao/perf12_20261105/gemm/repo` | `a3ec99fa5f3e820e` | `8a9a3a5435bfd0be` | `2e38f75a19defa22` |
| p13 repo (final evidence tree) | `/data41/audreymao/perf13_20261105/ew/repo` | `0830aed0e8062b88` | `8a9a3a5435bfd0be` | `07ffac67efb1af49` |
| `e74549f` (committed) | git | `efe423bea19ab6ab` | `aaaa8651db1beb6c` | `6a62508170ce28de` |

The `dd01d63` `exact_gemm.py` hash matches the candidate hash in `experiments/perf/p10_exact_gemm/RESULTS.md`.

## Final tree compared with e74549f

The p13 tree is not byte-identical to `e74549f`.

- `exact_gemm.py`: p13 `8a9a3a5435bfd0bebdc6ecf704f3591ef5c55d278e8f5322314d1eb24fedda65`; `e74549f` `aaaa8651db1beb6c4a1fc5484b550e483b1332c8a95485496cbc237e50257bd5`. The code is the same: the two ASTs are equal after removing docstrings. The differences are the module docstring text, one comment line that exists only in `e74549f` (`# Holdout keeps qkv M <= 16 ...`), and line wrapping of several calls.
- `exact_fuse.py`: p13 `07ffac67efb1af491cdbd96ad26a94bbb0d3bfdcc69cca4db69b2750c93beb3a`; `e74549f` `6a62508170ce28def3fe20ec7d5ba4e081797024de1016315318bf5404b7a82c`. The two files differ in one docstring line (line 15). The ASTs are equal after removing docstrings.
- Consequence: all numbers in this report were measured on the p13 tree. `e74549f` itself was not executed. The equal code makes the same behavior expected, but this is not verified on a GPU here.

## Round details

### p11: stacked integer GEMM (`exact_gemm.py` only)

- Adds `_stacked_linear`. It stacks the three activation limbs and three weight limbs into one int32 GEMM. The partial sums fit int32 for K <= 14336 and |limb| <= 128, as stated in the module docstring. The epilogue promotes to int64, adds the sparse residual, and rounds once with the same overflow check.
- Gate on `sm90()`, `0 < M <= 128`, and the linear's shape.

| Shape (N, K) | First variant (fa71) | Guarded variant (5cf8) |
|---|---|---|
| (4096, 14336) | all M | all M |
| (6144, 4096) | M >= 9 | M >= 17 |
| (28672, 4096) and (128256, 4096) | M >= 17 | M >= 17 and M != 64 |

The `exact_fuse.py` file is unchanged from `dd01d63` in p11. The reason for the guarded gate change is not recorded in the evidence files.

### p12: reuse of exact SiLU products (`exact_fuse.py` only)

- Adds `_silu_cached_kernel`. For 8B shapes (`FF == 14336`) with `0 < N <= 128`, the first pass stores the exact SiLU products in a scratch buffer, and the second pass reloads them. Other shapes keep the previous kernel.
- `exact_gemm.py` is unchanged from the guarded tree.

### p13: integer reciprocal division (`exact_fuse.py` only)

- `_norm_kernel` gains a `RECIP` constant. It is enabled for `D == 4096` and `M <= 128`, with 16 warps.
- Adds `_attn_recip_kernel`, used by `attn_out` for `K2 == 4096`, `HD == 128` and `0 < N <= 128`.
- `exact_gemm.py` is unchanged from p12.

## Correctness, item by item

Every row lists counts read from the output files. "PASS" means the script exited 0 and its own assertions held. The 8B rows use `const_8b_public.pt`. The 1B rows use `exact.pt`.

| Check (script, output) | p11 first (fa71) | p11 guarded (5cf8) | p12 (a3ec) | p13 (0830) |
|---|---|---|---|---|
| GEMM pytest `tests/test_vllm_exact_gemm.py`, 8B | 1 passed, 1 skipped | 1 passed, 1 skipped | not run (GEMM unchanged from guarded tree) | not run (GEMM unchanged from p12) |
| Fuse pytest `tests/test_vllm_exact_fuse.py`, 8B | not run | not run | 7 passed, 7 skipped | 7 passed, 7 skipped |
| Constructed edge cases | not run | `p11_edges.py`: PASS 48 cases (16 with reference overflow, 24 with checked flag) | `p12_silu_edges.py`: PASS 306 cases | `p13_recip_edges.py`: PASS 1989 cases (1836 norm, 153 attention) |
| Boundary kernel check, 8B, 85 cells | 85 cells, script equality asserts passed; 6 cells below 0.97 | 85 cells, script equality asserts passed; 7 cells below 0.97 | not run | not run |
| Held-out boundary check, 8B, 85 cells (GPU 3) | 85 cells; 3 cells below 0.97 | 85 cells; 0 cells below 0.97 | not run | not run |
| Boundary check, 1B, 85 cells | 85 cells, ratio range 0.988 to 1.005 | not run | not run | not run |
| Reference vs plugin, `check88`, 8B, 192 target rows | compare0 and compare1: 192 of 192 matched, diff 0 | 192 of 192 matched, diff 0 | 192 of 192 matched, diff 0 | 192 of 192 matched, diff 0 |
| Batch invariance B=1/32/128, eager 0 and 1 | BATCH_INV PASS (192 rows) | BATCH_INV PASS (192 rows) | BATCH_INV PASS (192 rows) | BATCH_INV PASS (192 rows) |
| Production token dumps, 8B, 128 requests x 48 tokens, base vs repo | mixed, greedy, logprobs: 6144 of 6144 tokens equal in each mode | same, guarded tree | same, base guarded tree | same, base p12 tree |
| Full invariance suite, 1B (`run_tests.sh`) | 14 runs (10 short, 4 long, the 2 references included), each compared to its reference: token diff 0, missing 0, logit-hash diff 0, inconsistent repeats 0; compare exit 0 | not run | same 14 runs, all zero; compare exit 0 | same 14 runs, all zero; compare exit 0 |

Details for the rows above:

- `check88` reference rows: 192 (4 prompts x 48 positions). Both eager modes match the reference on every row.
- Batch invariance (`batch_inv.py`, script assert `invariant_32_128`): the 4 target requests produce identical tokens and identical 192 per-step logits-hash rows at B=1, 32 and 128. The script prints PASS in all eight runs (4 trees x eager 0 and 1). I re-checked the token files for all eight runs directly; they are equal across B. I did not re-derive the hash rows from the `.hash` files.
- Full invariance suite, 1B: 12 short targets x 48 positions = 576 positions, plus 3 long targets x 16 positions = 48 positions. Each run has zero token and logit-hash differences. Forced-preemption runs report 5 preemptions in p11 (fa71), p12 and p13 logs.
- Production tokens: 128 requests of 64 input and 48 output tokens (`tokens.py`). The mixed and logprobs runs use temperatures cycling through 0, 1, 0.7, 0.5, 1.5 and 2 (logprobs run also sets `logprobs=1`). The greedy run uses T=0 for all requests. Each run has 6144 output tokens.
- Cross-round token check, read directly from the dumps: p11 base (`dd01d63`) equals p13 repo (`0830`) for mixed, greedy and logprobs; p11 base equals the p11 guarded tree; the guarded tree equals p12 repo. All comparisons are equal.
- Supporting micro-probes (not used for any headline claim): p12 `coeff.json` 20/20 equal, `epi.json` 30/30, `persist.json` 42/42, `sparse.json` 27/27, `silu.json` 72/72. p13 `norm.json` 45/45 equal, `attn.json` 54/54, `config.json` 24/24, `primitive.json`: signed_qr 16474 cases, round_recip 16684 cases, round_skip 16684 cases, all equal.
- The pytest skips are the `int8` parametrization of the shared `const` fixture. `BITEXACT_INT8_CONST` was not set. This matches the note in the p10 RESULTS.md.

## End-to-end throughput, 8B exact

Protocol (`multibench.py`): one vLLM engine per run with `max_num_seqs=128`, `max_model_len=2048`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.5`, seed 0. Batch sizes 1, 32 and 128 run in that order in one engine. Each batch gets two warm-up generations, then one timed generation of B random 128-token prompts, 128 output tokens, temperature 0, `ignore_eos`. Five repetitions. Base and repo alternate order by repetition parity. The driver waits for 0 MiB used and no compute process on the GPU before each run.

Values are output tokens per second: median [min, max], n=5. The change is the ratio of medians minus one, computed from unrounded values.

| Round | Base | Repo | B | Base tok/s | Repo tok/s | Change |
|---|---|---|---:|---:|---:|---:|
| p11 first (dd01 to fa71) | dd01d63 | fa71 | 1 | 64.3 [60.8, 69.9] | 65.9 [62.4, 70.0] | +2.6% |
| p11 first | dd01d63 | fa71 | 32 | 1537.4 [1489.5, 1561.8] | 1682.5 [1262.2, 1796.0] | +9.4% |
| p11 first | dd01d63 | fa71 | 128 | 3107.5 [2862.6, 3126.2] | 3278.1 [2543.6, 3544.8] | +5.5% |
| p11 guarded (dd01 to 5cf8) | dd01d63 | 5cf8 | 1 | 71.0 [69.1, 71.1] | 72.1 [71.2, 72.8] | +1.6% |
| p11 guarded | dd01d63 | 5cf8 | 32 | 1576.7 [1409.7, 1582.0] | 1815.0 [1799.8, 1838.7] | +15.1% |
| p11 guarded | dd01d63 | 5cf8 | 128 | 3153.9 [3079.7, 3159.3] | 3618.5 [3610.1, 3630.6] | +14.7% |
| p12 (5cf8 to a3ec) | 5cf8 | a3ec | 1 | 72.3 [72.2, 73.2] | 77.5 [76.3, 77.6] | +7.3% |
| p12 | 5cf8 | a3ec | 32 | 1826.8 [1825.8, 1844.6] | 1900.8 [1898.8, 1918.0] | +4.0% |
| p12 | 5cf8 | a3ec | 128 | 3619.9 [3617.3, 3641.5] | 3706.0 [3687.5, 3715.3] | +2.4% |
| p13 (a3ec to 0830) | a3ec | 0830 | 1 | 77.5 [76.5, 77.6] | 81.8 [81.6, 83.2] | +5.5% |
| p13 | a3ec | 0830 | 32 | 1903.1 [1899.5, 1905.7] | 2004.2 [2003.1, 2008.5] | +5.3% |
| p13 | a3ec | 0830 | 128 | 3701.9 [3692.0, 3717.3] | 3789.6 [3781.7, 3807.7] | +2.4% |

Notes:

- The min-max ranges of base and repo do not overlap in any p11 guarded, p12 or p13 row. They overlap in all three p11 first-variant rows. In that variant, the repo minimum at B=32 (1262.2) and at B=128 (2543.6) come from single low runs. No run was excluded.
- The p11 final tree appears to be the guarded variant (5cf8). It is the base of p12, and its files are the newest in the p11 directory. The first variant (fa71) is an earlier candidate in the same round. This is inferred from the directory and file history.
- Cross-session check: the p12 base (guarded tree) at B=128 gives 3619.9, and the p11 guarded repo, which is the same tree, gives 3618.5, in a different session.
- These numbers do not share the protocol of the p10 table (`bench.py`, one warm-up, one engine per run). Compare the rounds with each other, not with p10.
- No throughput was measured against `6faf8ef` or `e74549f` in a single session.

## Anomalies and gaps

1. `out/bench8` (p11, `bench_runner.py`) is incomplete. Its `result.json` has status RUNNING. Only `base_b32_r0` and `repo_b32_r0` finished. `base_b128_r0` started and has no result. The round uses `out/multibench8` instead. The partial run is not used.
2. p11 guarded edge check: the first run of `p11_edges.py` failed `assert any(r['reference_overflow'] ...)` at line 39 (log `out/guarded-edges-first.log`, 05:27:58). A later run passed with 48 cases (`out/guarded-edges.log`, 05:28:33). The script on disk has the same modification time as the failing run and still contains the failing assert. The cause of the first failure is not recorded and was not investigated. The 48-case result comes from the passing run.
3. Kernel-timing ratios below 0.97 (the rule from the p10 RESULTS.md, every final cell at or above 0.97). These are single measurements of three alternating CUDA graph pairs per cell:
   - p11 first, correct run: wqkv M=9 0.901; wgu M=64 0.952; wgu M=512 0.701; wd M=129 0.780; wd M=256 0.880; lm M=256 0.967.
   - p11 first, held-out run: wqkv M=9 0.908; wgu M=64 0.960; lm M=64 0.913.
   - p11 guarded, correct run: wgu M=129 0.940; wgu M=256 0.903; wd M=129 0.885; wd M=512 0.939; lm M=16 0.838; lm M=96 0.934; lm M=127 0.969.
   - p11 guarded, held-out run: no cell below 0.97 (minimum 0.996).
   - The lm M=16 cell in the guarded run uses the table path on both sides, because the stacked path needs M >= 17, and the table configs are identical in every cell. The ratio of 0.838 is therefore attributed to timing noise. This is an inference; the cell was not re-measured with more pairs.
   - Several sub-0.97 cells are on the stacked path (for example lm M=96 and wd M=512 in the guarded run). No cell was re-run.
4. 1B end-to-end throughput was not measured in p11, p12 or p13.
5. The 1B checks do not exercise the changed code. The new gates match 8B shapes only: stacked GEMM for N, K in {(4096, 14336), (6144, 4096), (28672, 4096), (128256, 4096)}; the SiLU cache for FF == 14336; the reciprocal norm for D == 4096; the reciprocal attention output for K2 == 4096. 1B has D = 2048, FF = 8192 and K2 = 2048. So the 1B boundary run (first p11 variant only) and the 1B invariance suites (p11, p12, p13) test the unchanged paths. They are not evidence for the new kernels. The 8B checks (`check88`, `tokens8`, `fuse8` edges, held-out boundaries) are the evidence for the new paths.
6. p12 `gemm.json`: 30 of 36 probe configurations were excluded with "out of resource: shared memory" errors. Six configurations completed. The probe is not used for any claim.
7. Invocation arguments for `check-runner.py`, `unit.py` and the probe scripts are not recorded in the files. The mode and size are inferred from output directory names (`check88`, `correct8`, `fuse8`). The GPU index is taken from the scripts where it is hard-coded.
8. p13 probe outputs (`norm.json`, `attn.json`, `primitive.json`, `config.json`, `recip-edges.json`) were written by scripts in the repo tree (`p13_probe.py`, `p13_attn_check.py`, `p13_recip_edges.py`). Output-path mapping is inferred from the script arguments.
9. Tarballs on host 141 (`*.tar.gz`) were not used as the primary source. The p13 `final-evidence.tar.gz` was spot-checked: six files (`fuse8/edges.log`, `fuse8/kernels.log`, `check88/compare0.log`, `check88/compare1.log`, `tokens8/result.json`, `multibench8/result.json`) are byte-identical to the `out/` copies. The other tarballs were not compared.
10. The `ah` harness `.agent-harness/POLICY.yaml` lists only the two `src` files as allowed paths. This commit adds only files under `experiments/perf/p11_p13_exact/`. No `ah start` or `ah verify` was run.

## Scope of evidence

What the evidence supports:

- Exact-mode 8B outputs (tokens, logits hashes, integer kernel outputs) are unchanged across the three rounds, on the H800 GPUs and runs listed below.
- Throughput for 8B exact at batch sizes 1, 32 and 128, T=0, 128-token inputs and outputs, five paired runs per cell, on idle GPUs.

What the evidence does not support:

- Any 1B throughput statement.
- Any int8-mode statement. Int8 mode was not run.
- Throughput for other shapes, other batch sizes, prompt lengths, or temperatures above 0. The production token dumps use temperatures above 0, but no throughput was measured for them.
- That `e74549f` itself was run. The p13 tree is code-equal to it; the byte-level differences are listed above.
- Behavior under concurrent GPU load. All throughput runs wait for an idle GPU.

## Reproduction

Host: `audreymao@10.154.162.141` (8 x NVIDIA H800). Interpreter: `/data41/audreymao/a17_vllm/venv/bin/python` (from the `result.json` command lines). vLLM 0.32.0 (from the engine log header).

Models and constants:

- 8B exact: `/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel`. Its `config.json` points to `bitexact_const` = `/data41/audreymao/xhw8b/out/const_8b_public.pt`. The constants file has `lm.limbs`, which selects exact mode in `model.py`. Weights are BF16 safetensors.
- 1B exact: `/data41/audreymao/perf7_20261105/exactplugin/exactmodel`, constants `/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt`.

Round directories (each contains `out/`, `repo/`, and the runner scripts that produced it):

- `/data41/audreymao/perf11_20261105/gemm/` (`repo`, `guardrepo`, `out`)
- `/data41/audreymao/perf12_20261105/gemm/` (`repo`, `out`)
- `/data41/audreymao/perf13_20261105/ew/` (`repo`, `out`)

Scripts copied into this directory (`experiments/perf/p11_p13_exact/`), by round:

| Copied script | Output directory on 141 | Trees (base, repo) | GPU | Note |
|---|---|---|---|---|
| `p11/multi_driver.py` with `p11/multibench.py` | `p11/out/multibench8` | dd01d63, fa71 | 2 | 5 reps, batches 1/32/128 |
| `p11/bench_runner.py` | `p11/out/bench8` | dd01d63, fa71 | 2 | incomplete, see anomaly 1 |
| `p11/guarded-driver.py` with `p11/multibench.py` | `p11/out/guardedbench8` | dd01d63, 5cf8 | 0 | 5 reps, batches 1/32/128 |
| `p11/check_runner.py` | `p11/out/correct8`, `correct1`, `check88` | fa71 | CLI | `check` and `correct` modes |
| `p11/guarded-check-runner.py` | `p11/out/guardedcorrect8`, `guardedcheck88` | 5cf8 | CLI | |
| `p11/holdout.py` | `p11/out/holdout8.json` | dd01d63 base, fa71 | 3 | held-out boundary check |
| `p11/guarded-holdout.py` | `p11/out/guardedholdout8.json` | dd01d63 base, 5cf8 | 3 | held-out boundary check |
| `p11/production.py` | `p11/out/tokens8` | dd01d63, fa71 | 1 | |
| `p11/guarded-production.py` | `p11/out/guardedtokens8` | dd01d63, 5cf8 | 1 | |
| `p11/suite.py` | `p11/out/fullsuite1` | fa71 (1B) | 0 | `tests/vllm/run_tests.sh p11` |
| `p12/multi-driver.py` with `p12/multibench.py` | `p12/out/multibench8` | 5cf8, a3ec | 0 | |
| `p12/check-runner.py` | `p12/out/check88` | a3ec (repo) | CLI | |
| `p12/unit.py` | `p12/out/fuse8` | a3ec | 1 | runs fuse pytest and `p12_silu_edges.py` |
| `p12/production.py` | `p12/out/tokens8` | 5cf8, a3ec | 3 | |
| `p12/suite.py` | `p12/out/fullsuite1` | a3ec (1B) | 2 | |
| `p13/multi-driver.py` with `p13/multibench.py` | `p13/out/multibench8` | a3ec, 0830 | 0 | |
| `p13/check-runner.py` | `p13/out/check88` | 0830 (repo) | CLI | |
| `p13/unit.py` | `p13/out/fuse8` | 0830 | 1 | runs fuse pytest and `p13_recip_edges.py` |
| `p13/production.py` | `p13/out/tokens8` | a3ec, 0830 | 3 | |
| `p13/suite.py` | `p13/out/fullsuite1` | 0830 (1B) | 2 | |

`multibench.py` is identical in p11, p12 and p13. `multi_driver.py` and `multi-driver.py` are the same script with different tree paths.

Scripts that the runners call, which are in the trees and were not copied here (copies are on the host):

- `experiments/perf/p10_exact_gemm/check_configs.py`. Identical in all trees and in `e74549f`.
- `experiments/perf/p10_exact_gemm/p11_edges.py`, `p12_silu_edges.py`, `p13_recip_edges.py`.
- `experiments/perf/p7_exactplugin/ref8.py`, `batch_inv.py`, `batch_inv_raw.py`, `cmp8.py`.
- `experiments/perf/p4_merge/tokens.py`.
- `tests/test_vllm_exact_gemm.py`, `tests/test_vllm_exact_fuse.py`, `tests/vllm/run_tests.sh`.

Reproduce a round by copying the round's runner into `/data41/audreymao/<round dir>/gemm/` (or `ew/`), with the tree directories in place, and running it with the same `CUDA_VISIBLE_DEVICES` as listed. Runners that take a CLI argument (`check-runner.py <mode> <gpu> <size>`, `guarded-check-runner.py`) need `mode` (`check` or `correct`), `gpu` and `size` (`1` or `8`). Each runner refuses to run if its output directory already exists, so remove the old output before a rerun.

Result files to compare against: `out/multibench8/result.json` (source hashes and command list), `out/check88/compare0.log` and `compare1.log`, `out/tokens8/*.json`, `out/fuse8/edges.log`, `out/fullsuite1/suite.log`.
