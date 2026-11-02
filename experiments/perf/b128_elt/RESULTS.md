# perf/elt2 (MLP silu/requant and logits/greedy in Triton): verification and throughput

## Conclusion

1. **Bit-identity: PASS.** All 14 plugin invariance runs of perf/elt2 (ded5391) give token_seqs_diff 0, missing 0, logit_hash_diff 0 against `run_ref`. All 14 `repo_X.hash` files are byte-equal (`cmp`) to the matching 29e5cb5 run made in the same session, and also to the 29e5cb5-code hashes from the perf/combo verification (`speedup_20261030/combo/out/suite`). GPU pytest: 20 passed, 1 skipped.
2. **Throughput at T=0, same GPU (141 GPU 7), two passes with base and repo alternating:** B=128 rises **+7.4%** (19,729 → 21,196 tok/s, mean of two passes; per-pass +8.5% and +6.4%). B=32 rises +3.8% (per pass +7.2% and +0.5%, so the change is within noise). B=1 is unchanged (−1.4%, within noise).
3. **Where the time went (B=128 decode profile):** wall 5.67 → 5.37 ms/step. Kernel sum 5.55 → 4.97 ms. The integer elementwise work, counting the new kernels (bucketed as `other`), went from 1.81 to 1.64 ms. Logits + sampling went from 0.93 to 0.57 ms. Attention and GEMM are unchanged.
4. **Long context, S=16k (single run):** decode step 3.169 → 3.080 ms at B=1 (−2.8%) and 11.830 → 11.644 ms at B=8 (−1.6%).
5. The B=128 gap to vLLM bf16 (`VLLM_BATCH_INVARIANT=1` with TRITON_ATTN, 34,941 tok/s in the combo session, **not re-measured here**) narrows from 0.56x to about 0.61x. Attention (1.58 ms) and the int8 GEMMs (1.08 ms) are now the largest buckets, and both lie outside this task.
6. Fusing the requant into a Triton int8 GEMM epilogue was **dropped**. In the microbenchmark it is slower than `torch._int_mm` plus a separate post kernel at every shape that matters (M=128: gate_up 96 vs 50 us, down 42 vs 24 us, lm head 320 vs 315 us). The file is kept untracked and is not part of the branch.

## What changed (ded5391 vs 29e5cb5)

- `vllm_plugin/tri_mlp.py` (new) replaces the ops between the MLP's gate_up GEMM and its down GEMM (`segs.post` → `silu_mul` → `pre_plain` → `stack_limbs`) with two Triton kernels. `_h_kernel` computes h = rsh(silu(g)·u, FR) together with the per-chunk max |t|, |sg|, |u| and |h|. `_quant_kernel` computes the row's quant_rows / rdiv32 / norm24 and writes both int8 limbs. Inductor had used four kernels for this and passed int64 [M, 2FF] tensors through memory. It is used only for CUDA with 2 limbs. `BITEXACT_TRI_MLP=0` restores the old path.
- `vllm_plugin/tri_logits.py` (new) does one pass over the int64 logits. It writes the float copy (`li.float() * 2^-16`) and the per-chunk (max, first index), then a second small kernel combines the chunks into the first argmax. `compute_logits` stores this argmax in `STASH.greedy`, and `integer_sample` uses it for T=0 rows in place of `intops.first_argmax`.
- Microbenchmark (`bench_elt.py`, GPU 1): at M=128, logits + greedy takes 419 → 72 us and the MLP middle takes 43 → 26 us.

## Integer-semantics review (verifier)

I read each new kernel against the reference ops (`int8/segs.py`, `int8/ops.py`, `intops.py`). No bound or rounding point moves:

- **post** (`_y`): acc = c0 + (c1 << 8) is formed in int64 from the int32 GEMM limbs, as in `join_limbs`. Then t = rsh(acc·m, fmsh) and y = rsh(t·mant, 24 − FR − e). These are the same int64 products in the same order. Wraparound mod 2^64 is identical, and the existing `post_check` flag (`amax(t) >= 2^38`) is still raised from the exact row maxima of |t|.
- **shifts:** `_shr` clamps the amount to 63 and `_shl` returns 0 for amounts ≥ 64. Both reproduce torch's int64 semantics for every amount, so `_rsh` equals `intops.rsh` for tensor shifts, including s ≤ 0.
- **silu:** same table index, clamp and interpolation. `(Δ·r + 2^(FR−FA−1)) >> (FR−FA)` is `rsh` with a constant positive shift (FR−FA = 4).
- **h = (sg·u + 2^(FR−1)) >> FR** is `rsh(sg*u, FR)`. The overflow flag `amax(sg)·amax(u) >= 2^62` uses the same int64 product of the same exact maxima.
- **quant_rows:** amax is the exact max of |h| over chunks and then clamped to ≥ 1. sh = bitlen(amax) − 30 uses `64 − clz` in place of the frexp-based `bitlen32`, and the two are equal for every x ≥ 1. as_, zs and the rdiv32 floor are followed by the same two exact int64 corrections. Since |zs·qmax| ≤ 2^30·32639 < 2^45 and d ≤ 2^31, the products q·d stay below 2^48, as in the reference. The float32 quotient therefore only needs to be within one unit, which the corrections remove.
- **norm24 and limbs:** `bitlen(as_) − 24` and `rsh`, then `((q+128)&255)−128` and `(((q−lo)>>8)+128)&255)−128`. This is `split_limbs` with n = 2, with the same padding to ≥ 24 rows (multiple of 8) and zeroed pad rows.
- **logits float copy:** int64 → float32 is round-to-nearest-even in both paths. The multiply by 2^-16 is exact (no subnormals for |l| ≥ 1).
- **first argmax:** each lane keeps its first strict maximum, and lanes and chunks are combined by integer max and then min index. This is order-free and returns the same index as `intops.first_argmax`, including all-tied rows and all-int64-min rows (unit test cases).
- **Scope:** `STASH.greedy` is set on every `compute_logits` call: to None for CPU tensors and to the new argmax for CUDA tensors. A stale value therefore cannot pair with new logits. Sampled rows (T > 0) do not use it. No attention file is touched.

These arguments are supported by the bit-exact unit tests `test_tri_mlp_silu_quant_exact` (M = 1/3/64/128/300, four scale/shift settings including post_check on and off) and `test_logits_argmax_float_exact`, and by the end-to-end suite below.

## Setup

- Server 141, **GPU 7** for every number below. GPUs 2–6 were not used. Trees: `/data41/audreymao/b128_20261031/elt/repo` (ded5391) and `.../elt/base` (29e5cb5), synced by `git archive`, with the untracked profiler tools copied in.
- Runner `experiments/perf/elt2/run.sh`, `STAGES="pytest suite bench longctx prof"`, outputs in `.../elt/out_full/`, 02:54–03:33 server clock. Verifier reruns and the second bench pass: `.../elt/verify/` (`rerun.sh`, `bench2.sh`), 03:33–03:41.
- Each launch waited until GPU 7 showed 0 MiB and no compute process. Another agent (the attn verifier) also used GPU 7 until 03:18. The bench, longctx, profile and verifier runs all started after 03:19 and had the GPU to themselves.

## Bit-identity

| check | result |
|---|---|
| GPU pytest (`test_gpu_kernels`, `test_vllm_kernels`, `test_int8_ops`, `test_decode_paths`, `test_overflow_check`) | 20 passed, 1 skipped |
| suite, repo vs `run_ref`: 10 short runs (12 targets × 48 positions) and 4 long runs (3 targets × 16) | 14/14 token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0 |
| `cmp repo_X.hash base_X.hash` (29e5cb5, same session), all 14 X | 14/14 byte-identical |
| `cmp repo_X.hash` against the combo-session 29e5cb5-code hashes | 14/14 byte-identical |

Three runs failed while the attn agent's jobs shared GPU 7: base_alone_pc0, repo_busy_pc0_s1 and repo_alone_vllmgraph. All three hit vLLM's start-up assertion "Error in memory profiling … current free memory" before producing any token. That failure blanked the runner's in-script compare for the short runs. I reran these three alone on GPU 7 (`out/verify/rerun.log`). All three pass compare and cmp. The other four short repo runs that the aborted compare skipped were compared separately (`out/verify/compare_rest.txt`, all 0).

## Throughput (output tok/s, `tests/vllm/bench.py`, 128 in / 128 out, GPU 7)

| B | T | 29e5cb5 pass 1 | elt2 pass 1 | 29e5cb5 pass 2 | elt2 pass 2 | mean ratio |
|---|---|---|---|---|---|---|
| 1 | 0 | 386.3 | 377.6 | 372.3 | 370.3 | 0.986x |
| 32 | 0 | 8,152.1 | 8,736.9 | 8,208.8 | 8,251.9 | 1.038x |
| 128 | 0 | 19,473.2 | 21,129.3 | 19,984.9 | 21,261.8 | **1.074x** |
| 1 | 1 | 367.3 | 373.3 | – | – | 1.016x |
| 32 | 1 | 7,910.6 | 8,052.7 | – | – | 1.018x |
| 128 | 1 | 17,481.7 | 18,342.1 | – | – | 1.049x |

- The mean ratio is the mean of elt2 divided by the mean of 29e5cb5. Pass 1 alternates base, repo per cell (runner). Pass 2 alternates repo, base (`bench2.sh`). The B=128 gain holds in both orders, at +8.5% and +6.4%. B=1 and B=32 vary between passes by about as much as the effect.
- 29e5cb5 measured 19.5–20.0k tok/s at B=128 here, against 19,354 in the README (different GPU and session).

## Long context (`tests/vllm/longctx_step.py`, S=16,384, CUDA-graph decode, one run each)

| B | 29e5cb5 ms/step | elt2 ms/step | change |
|---|---|---|---|
| 1 | 3.169 | 3.080 | −2.8% |
| 8 | 11.830 | 11.644 | −1.6% |

These are single runs, so differences under about 3% are not established.

## B=128 decode profile (`prof_vllm.py` + `buckets.py`, ms per step)

| bucket | 29e5cb5 | elt2 |
|---|---|---|
| wall (no profiler) | 5.669 | 5.368 |
| kernel sum | 5.551 | 4.970 |
| attention (`_acc_kernel` + `_max_kernel`) | 1.600 | 1.578 |
| int8 GEMM | 1.091 | 1.077 |
| elementwise (Inductor) | 1.805 | 1.294 |
| other (new `_h_kernel`, `_quant_kernel`) | 0.015 | 0.341 |
| logits + sampling | 0.933 | 0.573 |
| kv write | 0.101 | 0.101 |
| wall − busy | 0.126 | 0.400 |

Top decode kernels on 29e5cb5 (per step; the mapping to source op is inferred from kernel names and call counts and was not traced):

| ms | calls | kernel | source op |
|---|---|---|---|
| 1.133 | 16 | `_acc_kernel` | paged attention p·v (tri_paged) |
| 0.477 | 16 | cutlass s8 128x64 | MLP gate_up GEMM |
| 0.465 | 16 | `_max_kernel` | paged attention q·k max (tri_paged) |
| 0.460 | 32 | cutlass s8 64x64 | qkv and o GEMMs |
| 0.273 | 16 | `triton_red_fused_…frexp…` | MLP post + silu_mul + quant (now `_h_kernel` 0.253 + `_quant_kernel`) |
| 0.208 | 1 | cutlass s8 128x64 | LM head GEMM |
| 0.181 / 0.141 | 16 / 16 | `triton_red_fused_…div_floor…` | RMSNorm + quant (pre_norm) for attention / MLP |
| 0.155 | 16 | cutlass s8 128x64 x6 | MLP down GEMM |
| 0.150 / 0.146 / 0.116 | 16 / 16 / 32 | `triton_poi_fused_…` | post, rope, limb stacking |
| 0.099 + 0.092 | 1 + 1 | `elementwise_kernel`, `triton_poi_…` | logits float copy, argmax / LM head post |

- The kernel sum drops by 0.58 ms, but the wall time drops by only 0.30 ms. Wall − busy grows from 0.13 to 0.40 ms. The cause is UNKNOWN (not investigated). Possible causes include extra launches (two Triton launches plus a small `rmax.amax` reduction per layer) and the new allocations in `silu_quant`.

## Caveats

- vLLM bf16 BI was not re-measured on GPU 7. The 0.61x figure divides by the combo-session number (34,941 tok/s).
- Each long-context step time and each T=1 bench cell is a single run.
- The profile's top-kernel source mapping is inferred from names, not verified against the trace stacks.
- The fused int8 GEMM + requant epilogue (`fused_mm.py`, `bench_mm.py`, `test_fused_mm.py`) stays untracked in the worktree. It is slower and is not on the branch.

## Files (local `out/` is gitignored; copies kept in the worktree and on 141)

- `out/SUMMARY_pytest_suite_bench_longctx_prof.txt`, `out/go_pytest_suite_bench_longctx_prof.log`: the runner's summary and log.
- `out/bench/RESULT_lines.txt`: all bench RESULT lines (both passes). `out/longctx/*.log`, `out/prof/{base,repo}/summary.md`, `out/pytest_tail.txt`.
- `out/verify/`: verifier rerun and second-pass scripts, their logs, and the compare of the four remaining short runs.
- Large outputs (hash files, traces) stay on 141 under `/data41/audreymao/b128_20261031/elt/{out_full,verify}`.
