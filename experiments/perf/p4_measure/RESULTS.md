# p4_measure: reference numbers on main 6d42e87 (no source changes)

## Conclusion

1. 8B at B=128 is GEMM- and fixed-overhead-bound, not host-bound. Decode step 14.5 ms (T=0): int8 GEMM 6.5 ms (45%), elementwise 2.5 ms, attention 2.7 ms, other 1.1 ms, sampling/logits 0.5 ms. Wall minus kernel-busy is 1.16 ms. Kernels per step: 937 (the unfused path, 32 layers).
2. Two kernels are 49% of kernel time at B=128: cutlass `s8_128x64_128x3` (4.36 ms, 96 calls/step) and `_dec_acc_kernel` attention (2.68 ms, 32 calls). The cutlass `128x128_128x5` GEMM adds 2.11 ms (32 calls, likely down_proj).
3. 8B bench (5 reps): plugin 151 / 3,981 / 8,202 tok/s at B=1/32/128 (T=0). vLLM stock 151 / 4,508 / 13,931. VLLM_BATCH_INVARIANT=1 + TRITON_ATTN 50 / 1,748 / 10,578. Plugin is 59% of stock and 78% of BI+TRITON at B=128. T=1 costs 0-2% (B=32 -1.4%, B=128 -2.3%).
4. Speculative decoding (ngram k=4) hurts at B=32 and B=128 in every cell, on both workloads, at both temperatures (1B, 3 reps). Ratio k=4/k=0 from 0.98 (rep B=32 T=0) down to 0.38 (bench B=128 T=0.7). k=2 at B=32 is better than k=4 but still below k=0 in all four cells (best 0.97, rep T=0). Spec only pays off at B=1 (p3_merge: 1.68-1.94x on rep). Default must stay k=0 for batched serving.
5. Bit identity: token identity k=0 vs k=4 PASS in all 10 cells (12 comparisons incl. k=2), tokens only (no logit hashes), compared on rep 1 of 3. No code was changed; this lane is measurement only.

Setup: GPU 7 of 141 (H800), one process at a time, 132 launches. Before every launch GPU 7 showed 0 MiB and no compute process (`quiet_check.log`, same format as p3_merge). Host 1-min load at launch: min 7.7, median 51, max 283 (192-core shared box; the other GPUs on 141 were busy with other jobs). 132/132 runs rc=0. Tree: `git archive 6d42e87` + profiler from bitexact-llm/experiments/perf/profile. Runner `run.sh`, aggregator `agg.py`. Logs on 141: /data41/audreymao/p4_measure/out.

## 1. 8B profiler, decode ms/step (CUDA graph, 128 in / 128 out, steps 8-23 profiled, median of 3)

| B | T | wall | kernel sum | elementwise | attention | int8 GEMM | sampling/logits | other | wall - kernel busy | GPU idle in span | kernels/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0 | 14.495 | 13.339 | 2.540 | 2.678 | 6.475 | 0.525 | 1.116 | 1.158 | 0.544 | 937 |
| 128 | 1 | 14.744 | 13.695 | 2.541 | 2.673 | 6.469 | 0.909 | 1.113 | 1.050 | 0.557 | 939 |
| 32 | 0 | 7.515 | 6.957 | 1.919 | 0.750 | 3.388 | 0.427 | 0.465 | 0.559 | 0.560 | 937 |
| 32 | 1 | 7.618 | 7.045 | 1.920 | 0.751 | 3.390 | 0.516 | 0.465 | 0.574 | 0.570 | 939 |

Notes: wall = host step period (unprofiled pass, median); kernel sum = sum of kernel durations (buckets: gemm_int8, elementwise, attention, sampling_logits, other; kv_write, memcpy, gemm_bf16 about 0). T=1 adds 0.38 ms at B=128 in sampling/logits (0.52 -> 0.91), wall +0.25. Elementwise is fixed 1.9 ms even at B=32: those are the unfused quant/rescale triton kernels (fusion currently 1B geometry only), so they are a larger share at small B (26% of kernel time at B=32 vs 19% at B=128). Profiler tok/s column (8,226 at B=128 T=0) is within 0.3% of bench.

### Top 10 kernels, B=128 T=0 (ms/step, calls/step)

| ms | calls | bucket | kernel |
|---|---|---|---|
| 4.363 | 96 | gemm_int8 | cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16 |
| 2.676 | 32 | attention | _dec_acc_kernel |
| 2.108 | 32 | gemm_int8 | cutlass_80_tensorop_i16832gemm_s8_128x128_128x5_tn_align16 |
| 0.781 | 32 | other | _h_kernel |
| 0.335 | 64 | elementwise | triton_poi_fused___rshift____to_copy_add_bitwise_and_cat_div_floor_gt_le_mul_sub_where_3 |
| 0.330 | 1 | sampling_logits | cutlass s8_128x256_128x3 (lm_head) |
| 0.323 | 32 | elementwise | triton_red_fused (abs/amax/frexp quant) |
| 0.250 | 32 | other | _quant_kernel |
| 0.217 | 32 | elementwise | triton_poi_fused (frexp/clamp) |
| 0.203 | 32 | elementwise | triton_red_fused (abs/amax/frexp quant) |

The 96-call kernel is the three per-layer GEMMs that share the 128x64 tile (likely qkv, o_proj, gate/up as 3 calls each layer); the 32-call 128x128 is the remaining one. These all come from `torch._int_mm` (TABLE in `gemm_skinny.py` has only 1B shapes).

## 2. 8B bench, tok/s median [min-max], 5 reps, 128 in / 128 out, interleaved per rep in one session

| B | plugin T=0 | plugin T=1 | vLLM bf16 stock T=0 | vLLM BATCH_INVARIANT + TRITON_ATTN T=0 |
|---|---|---|---|---|
| 1 | 151 [148-152] | 151 [150-152] | 151 [151-151] | 50 [50-50] |
| 32 | 3,981 [3,881-4,135] | 3,927 [3,913-3,940] | 4,508 [4,487-4,517] | 1,748 [1,746-1,755] |
| 128 | 8,202 [8,180-8,257] | 8,012 [7,990-8,198] | 13,931 [13,800-14,159] | 10,578 [10,524-10,613] |

Plugin / stock: 1.00, 0.88, 0.59. Plugin / BI+TRITON: 3.0x, 2.3x, 0.78. Plugin T=1 / T=0: 1.00, 0.99, 0.98. Matches the p3_merge numbers within 1-3%.

## 3. Speculative decoding (ngram, prompt_lookup 2..5, k tokens), 1B, vLLM graphs, tok/s median [min-max], 3 reps

Tokens/s counts generated output tokens (4,096 at B=32, 16,384 at B=128).

| workload | T | B | k=0 | k=2 | k=4 | k=4/k=0 | k=2/k=0 |
|---|---|---|---|---|---|---|---|
| rep | 0 | 32 | 11,035 [10,900-11,657] | 10,685 [10,545-11,173] | 10,850 [9,881-11,225] | 0.98 | 0.97 |
| rep | 0.7 | 32 | 10,417 [10,349-10,533] | 7,140 [7,102-7,294] | 6,845 [6,688-6,903] | 0.66 | 0.69 |
| bench | 0 | 32 | 11,709 [11,522-11,852] | 8,265 [7,987-8,477] | 7,763 [7,554-7,895] | 0.66 | 0.71 |
| bench | 0.7 | 32 | 11,033 [10,842-11,085] | 6,574 [6,565-6,587] | 5,848 [5,723-5,861] | 0.53 | 0.60 |
| rep | 0 | 128 | 24,965 [23,374-25,353] | | 16,306 [14,714-16,454] | 0.65 | |
| rep | 0.7 | 128 | 22,083 [21,414-23,171] | | 11,160 [10,910-11,211] | 0.51 | |
| bench | 0 | 128 | 26,298 [25,514-26,882] | | 11,912 [11,713-13,903] | 0.45 | |
| bench | 0.7 | 128 | 25,251 [24,804-25,660] | | 9,631 [9,060-10,130] | 0.38 | |

rep = copy-heavy prompts (high ngram acceptance), bench = random-token prompts (low acceptance). Even on the copy-heavy workload spec does not win at B>=32: the verify step with k+1 rows per sequence multiplies GEMM rows (B=128, k=4: 640 rows) and the host sampler runs per row, so the step costs more than the accepted tokens save. k=2 beats k=4 in all four B=32 cells but never k=0 (closest: rep T=0, 0.97). Smaller k helps because verify width shrinks; it does not fix the batched regime. Acceptance counts were not extracted (RESULT `spec` field is empty on the k=0 rows; k>0 logs have it, not tabulated).

Token identity, k=0 vs k=4 (and k=2 at B=32), rep 1 of 3 per cell, `spec_identity.txt`: rep/bench x T 0/0.7 x B 32/128 k=4 all IDENTICAL (8/8); k=2 at B=32 IDENTICAL (4/4).

## Not done / not verified

- Logit-hash identity for spec (tokens only). Identity compared on rep 1 only; reps 2-3 not compared.
- No attribution of the spec slowdown (verify GEMM width vs host sampler vs scheduler) by profiler; the explanation above is inferred from the shapes.
- 8B profile steps 8-23 only (16 steps); prefill buckets recorded in the json but not tabulated.
- Host load was high on the shared box (median 1-min load 51); GPU 7 itself was empty at each launch. Spread across reps stays under 4% except plugin T=0 B=32 (3,881-4,135, 6%).
- Which of qkv/o_proj/gate-up/down maps to which cutlass kernel is inferred from call counts, not from kernel names with shapes.
