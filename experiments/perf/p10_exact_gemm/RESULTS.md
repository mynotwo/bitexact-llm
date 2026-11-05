# p10 exact GEMM: H800 configuration tuning

This change keeps the original bf16-weight exact mode and changes only `FusedLin.TABLE`. The arithmetic kernels, rounding, overflow flags, default dispatch and all configurations above 128 rows remain unchanged. The baseline is main `6faf8ef210762677d10a144a1188c329b9ac4d8c`; the candidate is branch `perf10/exact-gemm`.

## End-to-end throughput

Existing `tests/vllm/bench.py`: 128 input / 128 output tokens, one warm-up generation then one timed generation, prefix caching enabled, CUDA graphs, and five independent engine processes per cell. Base and candidate alternate on the same GPU, with the order reversed in odd repetitions. No observations are excluded. Values below are output tokens/s, median [min, max]. The percentage is the ratio of the two medians minus one.

| Model | Batch | Temperature | Baseline | Candidate | Change |
|---|---:|---:|---:|---:|---:|
| 1B | 1 | 0 | 254.6 [253.8, 256.6] | 254.6 [93.9, 255.0] | +0.0% |
| 1B | 1 | 1 | 249.7 [249.5, 254.5] | 249.5 [249.1, 250.4] | -0.1% |
| 1B | 32 | 0 | 6,777.2 [6,229.2, 6,795.2] | 7,007.1 [6,946.4, 7,064.3] | +3.4% |
| 1B | 32 | 1 | 6,363.1 [6,358.7, 6,372.1] | 6,512.9 [6,502.6, 6,569.2] | +2.4% |
| 1B | 128 | 0 | 13,862.4 [13,848.7, 13,878.9] | 14,162.4 [14,153.6, 14,229.5] | +2.2% |
| 1B | 128 | 1 | 12,479.1 [12,437.8, 12,499.5] | 12,711.9 [12,678.6, 12,811.5] | +1.9% |
| 8B | 1 | 0 | 65.4 [64.7, 65.5] | 70.9 [70.9, 70.9] | +8.4% |
| 8B | 32 | 0 | 1,357.8 [1,354.9, 1,364.2] | 1,579.4 [1,574.0, 1,581.3] | +16.3% |
| 8B | 128 | 0 | 3,085.6 [3,083.2, 3,093.7] | 3,160.9 [3,158.7, 3,173.1] | +2.4% |

The first 1B candidate batch-1/T=0 observation was 93.9 tokens/s; its log includes a Triton JIT warning. A later run also contains that warning while producing normal throughput, so the warning alone does not establish the cause of the slow observation. That observation remains in the table and five-run statistics. The batch-1 decode configuration of 1B is unchanged; no batch-1 speedup is claimed.

## Correctness

- 1B exact suite: all 14 runs on each tree match the standalone exact engine token-for-token and logits-hash-for-logits-hash. Short targets: 12 x 48 positions; long targets: 3 x 16 positions, with 5k/9k/14k input contexts. Prefix caching on/off, mixed batches, chunked prefill, forced preemption, eager execution and CUDA graphs are included. The forced-preemption logs report positive preemption counts.
- 8B exact: 4 x 48 positions match the independent reference in eager and CUDA graph modes; batch sizes 1/32/128 are invariant. Raw reference files are nonempty and contain exactly 192 target positions.
- Full GPU pytest: 147 passed, 8 skipped. The skipped cases are the existing exact-only tests parametrized with int8 constants. The separate 8B exact GEMM check passed (1 passed, 1 expected int8 skip). Each exact GEMM test covers 5 linear shapes x 16 row counts x 3 input/shift ranges, including overflow flags.
- Production token dumps without hash logging: base and candidate match for mixed temperatures, greedy sampling, and logprobs enabled; each dump has 128 requests x 48 tokens.
- Configuration checks: 80 shape/row cells per model, comparing the first five real layers for each non-head shape and the LM head. Both integer outputs and flags match. Each cell has three alternating CUDA graph timing pairs. Every final cell has base/candidate kernel-time ratio >= 0.97.

The first candidate extended two 8B single-row settings through row 8. The down projection at row 8 regressed from 89.08 to 109.35 microseconds. The final table uses both settings only at row 1; the complete boundary matrix and end-to-end checks were rerun after that correction.

## Decode profile

The existing profiler runs batch 128/T=0 with prefix caching disabled. These single profiles explain the work distribution; their throughput is not mixed with the five-run benchmark above. Decode windows cover steps 8-23. Times are milliseconds per step.

| Model | Tree | Host step median | GPU kernels total | `gemm_int8` bucket |
|---|---|---:|---:|---:|
| 1B | base | 8.341 | 8.11 | 4.9181 |
| 1B | repo | 8.134 | 7.9845 | 4.7226 |
| 8B | base | 38.274 | 37.6154 | 28.4944 |
| 8B | repo | 37.284 | 36.5681 | 27.5588 |

The profiler bucket name `gemm_int8` describes the limb-multiply kernels; the model still uses the original bf16-weight exact representation.

## Reproduction and evidence

Host: `audreymao@10.154.162.141`; environment: `/data41/audreymao/a17_vllm/env.sh`. 1B performance uses GPU 0, 8B performance GPU 2; GPU 1 is used only for correctness. The runner requires 0 MiB used and no compute process before each benchmark process and stage command; the 14-run correctness suite executes serially within its assigned GPU. GPU 4/5/6 are never used. Python 3.11.15, PyTorch 2.13.0+cu129, Triton 3.7.1, vLLM 0.32.0+cu129.

Remote root: `/data41/audreymao/perf10_20261105/exact_gemm`, with immutable `base`, candidate `repo`, and raw `out` directories. Local raw evidence is in `out/` beside this report and is git-ignored. `run.py` records every command, exit status, model path and source SHA-256. `verify_results.py` independently validates raw tokens, hashes, missing/extra positions, inconsistent recomputation, source provenance, full benchmark cells and profiles; it does not accept runner PASS alone.

The remote source export includes one macOS AppleDouble `._exact_gemm.py` sidecar. The verifier binds its exact bytes in `out/exact_gemm.appledouble` and independently checks every real source/test/runner file against `out/source_inventory.json`. No source file is dropped from the hash check.

Candidate exact_gemm.py SHA-256: `91bda728bda6240cee5b19130e04d0cb5abf25a820e4a21d2d6183b0c2e61eb4`.
Remote source-tree SHA-256 including the recorded sidecar: `49de9c272a9ffc885b4f7f6b04f90d5b1fcf1f910f1d3cdf5c1319eecee52093`.

Run `python3 experiments/perf/p10_exact_gemm/verify_results.py` after restoring the raw `out/` evidence. Use `run.py STAGE --tree base|repo --size 1|8 --gpu N --tag UNIQUE` on the remote host to generate new runs. Stages used here: `configs`, `suite`, `pytest`, `kernels`, `check8`, `tokens`, `bench`, `profile`. Tables and raw five-run measurements are also available in `out/final_summary.json`.

Scope of evidence: performance measurements are H800-only, for these 1B and 8B exact models. 8B long-context and other GPU architectures were not rerun in this task. The branch has not been merged into main.

## Harness status

The diff, declared GPU-evidence behavior check, scope, artifact, dependency and lint checks pass.
Formal completion remains blocked by one inherited native-patch observation
(`20261105T185614.288694Z-ede11546/ev_000010`): the file write was observed, but the host's patch response
format was not recognized. The current harness has no supported recovery command for replacing that
observation with the later successful file write and verification. The separate scp timeout was recovered
by successfully rerunning its original command. No harness ledger, policy or adapter was modified to
override this result; the overall harness status is PARTIAL, not READY_FOR_REVIEW.
