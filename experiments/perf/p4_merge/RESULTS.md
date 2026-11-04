# p4_merge: perf4/merge (fuse8b + gemm8b + measure) vs main 6d42e87

Branch perf4/merge, HEAD 1aa7875 plus one test-script fix (below). H800 on 141. Benches and profiles on GPU 7 (0 MiB, no compute process at every launch, `logs/quiet_check.log`; host 1-min load at launch: min 2.6, median 6.7, max 21). Correctness suites on GPU 0 and 3. Runner `run.sh`, aggregation `agg.py`, raw summaries in `logs/`.

## Conclusion
1. Bit identity: PASS in every check. The two lanes interact without issue: no failure, no fix needed in `src/`.
2. 8B tok/s, merge vs base (median of 5, interleaved): T=0 B=1 +14%, B=32 +19%, B=128 +20% (8,266 to 9,933). T=1 the same (+14 / +19 / +20%). The gains are close to the product of the two lanes (fuse8b alone +12.7%, gemm8b alone +8.4% at B=128: 1.127 x 1.084 = 1.22 predicted, 1.20 measured).
3. Versus vLLM in the same session (8B, T=0): plugin / stock = 1.15 (B=1), 1.06 (B=32), 0.70 (B=128); plugin / BATCH_INVARIANT + TRITON_ATTN = 3.5x, 2.7x, 0.93. The plugin now beats stock at B=1 and B=32 and is within 7% of BI+TRITON at B=128 (was 78%).
4. 1B unchanged: all 12 cells within noise (below). 8B long context S=16k decode: -11% (B=1), -6% (B=8).

## Bit identity
| check | result |
|---|---|
| `pytest tests/` with `BITEXACT_INT8_CONST` + `BITEXACT_INT8_CONST_8B` (GPU 0) | 126 passed, 2 skipped. Skips are `test_invariance.py` (needs BITEXACT_EXACT_CONST) and `test_overflow_check.py` (needs BITEXACT_VLLM_MODEL); neither touches these lanes. 8B tri_fuse (13 tests) and all gemm_skinny TABLE shapes ran. |
| 1B 14-run suite (10 x 12 targets x 576 positions + 4 long runs x 3 targets x 48 positions), merge vs base 6d42e87 (GPU 0 merge, GPU 3 base) | 14 of 14 `.hash` files cmp OK; token_seqs_diff 0, logit_hash_diff 0 in every run; compare_exit 0 and compare_long_exit 0 |
| p2_sample token dumps (128 req; mixed T 0/1/0.7/0.5/1.5/2; T=0; logprobs=1) | 3 of 3 IDENTICAL |
| 8B plugin vs standalone IntLlama reference (ref + 7 runs: alone, busy pc1/pc0, budget 64, eager, vLLM graph busy/alone), GPU 3 | compare_exit 0, 0 token and 0 logit-hash diffs, 576 positions each (`logs/s8_compare.txt`) |
| 8B batch_inv.py alone vs 32 vs 128, graphs and eager | BATCH_INV PASS, 192 rows each |
| spec_check k=0 vs k=4, 1B, B=1 (suite, rep and bench x T 0 / 0.7) | 5 of 5 IDENTICAL (tokens only) |

## 8B bench, tok/s, 128 in / 128 out, median [min-max], 5 reps interleaved in one session (GPU 7)
| B | T | base 6d42e87 | merge | delta | vLLM stock (T=0) | vLLM BI + TRITON_ATTN (T=0) |
|---|---|---|---|---|---|---|
| 1 | 0 | 153 [148-153] | 175 [174-181] | +14% | 152 [151-152] | 50 [50-51] |
| 32 | 0 | 4,028 [4,008-4,030] | 4,811 [4,772-4,905] | +19% | 4,526 [4,312-4,533] | 1,753 [1,748-1,758] |
| 128 | 0 | 8,266 [8,244-8,421] | 9,933 [9,925-9,968] | +20% | 14,097 [14,040-14,329] | 10,672 [10,670-10,788] |
| 1 | 1 | 152 [152-159] | 174 [174-175] | +14% | | |
| 32 | 1 | 3,969 [3,951-4,150] | 4,714 [4,691-4,738] | +19% | | |
| 128 | 1 | 8,056 [8,034-8,064] | 9,686 [9,662-9,783] | +20% | | |

## 1B bench, same session, median [min-max], 5 reps
| B | T | base | merge | ratio | vLLM stock | vLLM BI + TRITON |
|---|---|---|---|---|---|---|
| 1 | 0 | 528 [522-564] | 529 [525-533] | 1.00 | 610 | 215 |
| 32 | 0 | 13,222 [12,886-13,326] | 13,234 [13,078-13,593] | 1.00 | 16,028 | 7,529 |
| 128 | 0 | 29,771 [29,574-29,914] | 29,633 [29,575-29,966] | 1.00 | 33,888 | 34,453 |
| 1 | 1 | 517 [498-525] | 524 [523-529] | 1.01 | | |
| 32 | 1 | 12,393 [11,846-12,602] | 12,439 [12,268-12,522] | 1.00 | | |
| 128 | 1 | 28,789 [28,553-29,030] | 28,706 [28,583-28,987] | 1.00 | | |
1B is unchanged (all ratios 1.00-1.01, inside rep spread).

## 8B profile, B=128 T=0 decode, ms/step, median of 3 (prof_vllm.py, interleaved)
| | base | merge |
|---|---|---|
| step wall | 14.487 | 11.778 |
| kernel sum | 13.350 | 10.871 |
| gemm_int8 (cutlass `_int_mm`) | 6.481 | 2.900 |
| elementwise | 2.538 | 0.754 |
| attention | 2.683 | 2.622 |
| sampling / logits | 0.528 | 0.489 |
| other | 1.115 | 4.101 |
| wall - kernel busy | 1.139 | 0.908 |
| GPU idle in span | 0.545 | 0.248 |
| kernels per step | 937 | 388 |
The bucket `other` grows because the new Triton skinny GEMM kernels (`_k`, about 2.65 ms in the gemm8b profile) and the fused kernels (about 1.2 ms) are not classified as gemm_int8 / elementwise by `buckets.py`. GEMM plus other: 7.60 to 7.00 ms; elementwise saving 1.78 ms and kernel-count drop from 937 to 388 carry most of the gain. Remaining step: attention 2.6, cutlass gate_up 2.1 (no Triton win at 256 rows), Triton qkv/wo/wd 2.65, fused 1.2.

## 8B long context, decode ms/step (graph.decode_forward, `tests/vllm/longctx_step.py`), S=16384, median of 3
| B | base | merge | delta |
|---|---|---|---|
| 1 | 7.981 [7.976-7.986] | 7.107 [7.105-7.109] | -10.9% |
| 8 | 20.738 [20.735-20.748] | 19.467 [19.455-19.474] | -6.1% |
`longctx_step.py` had the 1B cache shape hard-coded (8 KV heads x 128). One-line fix on this branch: shape from `m.HKV, 2 * m.HD`; 1B values are unchanged by it. S=32k not run.

## Not done / not verified
- First attempt of the `spec` stage hit a missing module because I re-synced the tree while it ran (my error); the stage was rerun in full and the table above is that rerun. The first 1B `tokens` and bench stages were not affected.
- Spec logit hashes (tokens only), 8B spec, S=32k long context, 8B T=1 profile: not run.
- Triton skinny table tuned on H800 only.
- The profile's tok/s column is not comparable to the bench (profiling overhead).
