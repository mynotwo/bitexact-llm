# Integer elementwise speedup (branch perf/fuse-elementwise): independent verification

## Conclusion

1. **Bit-identity: PASS.** On the same GPU, the full plugin invariance suite (`tests/vllm/run_tests.sh`, short and long targets) shows 0 token_seqs_diff, 0 missing and 0 logit_hash_diff for all 14 runs of 1295e53 against its own reference. Branch logit hashes are byte-identical to the 48a60f7 hashes in all 14 hash files, `ref.hash` and `long_ref.hash` included. The GPU pytest set, including `tests/vllm/test_decode_paths.py`, passes (17/17).
2. **Throughput at T=0** (bench, second pass, GPU 0) is **1.21x / 1.25x / 1.42x** at B=1/32/128: 198.9 → 240.4, 4,586 → 5,725 and 9,320 → 13,247 tok/s. At T=1 it is 1.03x / 1.21x / 1.34x.
3. **Decode elementwise time per step is 46% lower at B=32** (2.49 → 1.33 ms) **and 63% lower at B=128** (4.96 → 1.81 ms). Decode kernels per step drop from 826 to 616. GEMM, attention and sampling time are unchanged, as expected, because those kernels were not touched.
4. **Three changes produce the gain. Integer semantics are unchanged in all three:** (a) the per-GEMM run-time overflow check, a global amax run 4x per layer, is replaced by a load-time worst-case bound that cannot fire on this model; (b) int64 division and float64 `frexp` in quantization are replaced by a float32 estimate followed by an exact int64 correction; (c) global amax reductions are reduced per row first, so Inductor parallelizes them. See "Diff review" below.

## Setup

- Server 141, **GPU 0** (NVIDIA H800, 0 MiB and no compute process before each launch). vLLM 0.32.0 (`/data41/audreymao/a17_vllm/venv`). Baseline 48a60f7 and branch 1295e53 were exported with `git archive` to `/data41/audreymao/speedup_20261030/fuse/{base_repo,repo}`. Runner: `experiments/perf/fuse/run.sh go`, run 06:14–06:57 server clock. Outputs are in `/data41/audreymao/speedup_20261030/fuse/cmp/{suite,bench,prof,hashes.txt,SUMMARY.txt}`.
- Baseline and branch runs alternate on the same GPU (suite: base then branch; bench: base and branch alternate per cell; profile: base and branch alternate per B).
- Bench: `tests/vllm/bench.py` with 128 input and 128 output tokens, `ignore_eos`, CUDA graph, `max_num_seqs=B`, timing from the second of two identical `generate` calls.
- Profile: `experiments/perf/profile/prof_vllm.py` and `buckets.py`, decode steps 8–23 (16 steps), T=0.

## Bit-identity

| check | result |
|---|---|
| suite, branch vs own `run_ref` (10 short runs, 12 targets × 48 positions each; 4 long runs, 3 targets × 16 positions) | all token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0; compare_exit 0, compare_long_exit 0 |
| suite, baseline 48a60f7 (same GPU, same session) | same, all 0 |
| branch runs vs **baseline** `run_ref` (`compare.py base_ref fuse_*`) | short_vs_base_exit 0, long_vs_base_exit 0 |
| hash files, `cmp` of `fuse_X.hash` vs `base_X.hash` for all 14 X (ref, long_ref, alone_pc0/1, busy_pc1_s0, busy_pc0_s1, busy_pc1_s2_b64, busy_pc1_s3_preempt, busy_eager_s4, busy_vllmgraph_s5, alone_vllmgraph, long_alone, long_busy, long_busy_eager) | 14/14 identical (ref md5 c670c0cb…, long_ref md5 67725430…) |
| GPU pytest of 1295e53 (`test_gpu_kernels`, `test_vllm_kernels`, `vllm/test_decode_paths`, `test_int8_ops`, `test_intops`), real constants | 17 passed |
| decode_forward micro bench hidden-state hash, base vs branch, B=1/32/128 | identical (76baf7b8…, eb686de3…, 326796946…) |
| independent stress test (this verifier): `rdiv32` vs `rdiv` on 40M `(a, b)` with b in [2^29, 2^30], \|a\| ≤ qmax(2)·b, a quarter at \|a\| = qmax·b, plus 4M exact ties ±2; `bitlen32` vs `bitlen` on 4M random values plus every 2^k ± 4 and every float32 rounding boundary 2^k − 2^(k−25) ± 4 | 0 mismatches on CPU, CUDA eager and CUDA `torch.compile` |
| `post_bound` on the real constants (all 16 layers × 4 linears + lm head, nl=2) | worst \|t\| bound = 2^32.74 < 2^38, so `post_check = False` and the run-time check is provably dead |

## Diff review (48a60f7..1295e53, `src/`)

- **`ops.bitlen32`** has the same structure as `intops.bitlen`, but its estimate is `float32` instead of `float64`. The conversion is monotone, and powers of two are exact in float32, so the estimate is k or k+1 and the int64 comparison removes the +1. For x ≥ 2^63 − 2^39 both versions round to 2^63 and behave identically. Used in `quant_rows`, `norm24` and `rms`.
- **`ops.rdiv32`** has the same formula, floor((2a+b)/(2b)). The float32 quotient has a relative error of about 3·2^-24, and \|n/d\| < 2^16 keeps the absolute error below 2^-6, so the floor is off by at most 1. The two exact int64 comparisons fix that error in both directions, and \|q·d\| < 2^17·2^31 cannot overflow. It is used only when qmax(nl) < 2^15, i.e. nl ≤ 2 (nl=2 here). The preconditions hold in `quant_rows`: rsh is monotone, so \|zs\| ≤ as_ < 2^30 and \|n\| < 2^46; nl ≥ 3 keeps int64 `rdiv`.
- **`post_bound` and `Segs(post_check)`**: \|q\| ≤ qmax follows from \|zs\| ≤ as_ and round-half-up. The bound qmax·Σ\|W\|·\|m\| plus monotone rsh is a valid upper bound, and an `assert` guards int64 overflow of l1·\|m\|. The check only set the error flag and never changed outputs. Other models whose bound fails keep it.
- **`segs._amax`, graph final flag**: per-row amax followed by amax over rows equals the global amax, because max is associative and the values are exact integers.
- **`attn_div`** moves `rdiv(num << (FR-FV), den)` from eager ATen into the compiled `_post` region. It is the same function; the 14 identical hash files and `test_decode_paths` (eager vs CUDA graph vs IntLlama) cover it. The attention kernel itself (`tri_paged.attend`) and the sampler are untouched.
- No change to rounding direction, saturation or shift direction. Activation dtype stays int64.

## Throughput (tok/s, bench second pass, GPU 0)

| T | B | 48a60f7 | 1295e53 | speedup |
|---|---|---|---|---|
| 0 | 1 | 198.9 | 240.4 | 1.21x |
| 0 | 32 | 4,586.3 | 5,725.2 | 1.25x |
| 0 | 128 | 9,319.6 | 13,247.1 | 1.42x |
| 1 | 1 | 210.4 | 216.9 | 1.03x |
| 1 | 32 | 4,238.5 | 5,139.5 | 1.21x |
| 1 | 128 | 8,290.1 | 11,118.4 | 1.34x |

Recomputed from `out_tokens / sec` in each RESULT line (e.g. 16,384 / 1.237 s = 13,245 tok/s; the 2 tok/s gap comes from the rounding of `sec`). The earlier det_compare baseline on GPU 7 was 223 / 4,319 / 9,269 (T=0). This run's baseline differs by −11% / +6% / +1%, so treat B=1 differences of ≈10% as run-to-run noise.

## Decode per-step buckets (ms per step, profiler, T=0)

| B | variant | wall | GPU busy | GPU idle (wall − busy) | gemm_int8 | elementwise | attention | sampling_logits | kernels/step |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 48a60f7 | 6.660 | 4.363 | 2.297 | 0.816 | 2.487 | 0.519 | 0.434 | 826.1 |
| 32 | 1295e53 | 5.301 | 3.225 | 2.076 | 0.816 | **1.332** | 0.525 | 0.441 | 616.1 |
| 128 | 48a60f7 | 12.845 | 8.890 | 3.955 | 1.092 | 4.959 | 1.591 | 1.126 | 827.1 |
| 128 | 1295e53 | 9.280 | 5.647 | 3.633 | 1.096 | **1.810** | 1.601 | 1.022 | 616.1 |

- At B=128 the elementwise bucket shrinks by 3.15 ms of the 3.57 ms wall saving. The rest is mostly a smaller idle gap (−0.32 ms); the sampler is unchanged and remains for the sampler task.
- Kernel names: the baseline's largest elementwise kernel (B=128: 1.022 ms, ×48/step, `triton_red_fused_..._amax_clamp_...`, consistent with the removed post-GEMM check plus its row quantization) and `triton_red_fused_amax_11` (0.309 ms) are gone. The largest remaining elementwise kernel is 0.274 ms ×16 (RMSNorm/quant reduction).
- Micro bench (`experiments/perf/fuse/micro.py`, decode_forward under a CUDA graph, GPU 0, earlier the same day): B=128 elementwise 3.02 → 2.42 ms (cf55c9e) → 1.87 ms (1295e53), elementwise kernels per step 698 → 522 → 474. The micro bench reports smaller gains than the vLLM profile (−1.15 vs −3.15 ms at B=128). **Unverified why**: the micro bench uses a fixed short context and a single decode function, while the profile includes the rest of the vLLM step.

## Bandwidth check (estimate, not measured with Nsight)

At B=128, the largest activation per layer is the gate/up output, [128, 16,384] int64 = 16 MiB. One read takes about 5 µs at the H800's 3.35 TB/s. In the baseline, the post-GEMM amax kernel took 1.022 ms / 48 = 21 µs per call. On the branch, the largest kernel takes 0.274 ms / 16 = 17 µs per call. These elementwise kernels are therefore not bandwidth-bound on int64 activations. They are dominated by int64 ALU work and division, plus serial reductions, which is consistent with which changes helped. Narrowing activations to int32 would save bytes, but the expected gain is small at these sizes. **Unverified**: no per-kernel DRAM-throughput counters were collected.

## Caveats

- One timed pass per bench cell and one profiler window per B, all on GPU 0. Other users' jobs were running on GPUs 1–7 during the run; no CPU-side contention measurement was taken.
- The suite runs at T=0 only. At T=1 only throughput was measured; token identity at T=1 was not checked separately (the sampler is not changed by this branch).
- The prefill buckets were not analyzed.
