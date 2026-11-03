# p2_merge: perf2/merge (attn + gemm + sample) against base 971a3e1

## Conclusion

1. **Bit-identity: PASS.**
   - GPU pytest on the merge: 27 passed, 1 skipped.
   - `run_tests.sh` 14 runs, base and merge: every run has token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0; compare_exit 0 and compare_long_exit 0. All 14 hash files are `cmp`-identical (14 cmp OK, 0 DIFF).
   - Production-path token dumps (128 requests; mixed temperatures 0/1/0.7/0.5/1.5/2; T=0 only; logprobs=1): all three IDENTICAL, base vs merge.
2. **Long context S=16k does not regress:** decode ms/step B=1 2.868 -> 2.379, B=8 7.706 -> 6.582.
3. **tok/s (median of 5, interleaved base/merge): merge is faster in every cell.** T=0: +17% (B=1), +16% (B=32), +16% (B=128). T=1: +18% (B=1), +6% (B=32), +3% (B=128, inside the noise). At B=128 T=0 the merge reaches 27.2k tok/s against 32.3k for vLLM bf16 BATCH_INVARIANT + TRITON_ATTN (84%); base was 23.5k (73%).
4. **The host was not quiet, so ranges are wide** (see Bench). Do not read the T=1 B=128 and B=32 gains as significant.
5. **No lane needs to be dropped on these numbers**: nothing is slower in any median. Per-lane attribution was not run on the merge (each lane's own RESULTS has its isolated numbers).

## Environment and quiet check

141, GPU 0 (H800), CPU Xeon 8457C, governor `performance` (so frequency scaling is not the cause). The host (192 cores) was running other users' jobs throughout: load average at launch ranged 13.7 to 146.6 (1 min; `out/quiet_check.log`, 101 launches), and from about 06:50 other jobs also took memory on GPU 0 between our runs (our guard waits for 0 MiB before each launch, so no run overlapped another GPU process). Two runs failed on a foreign job's memory (vLLM memory-profiling assertion): `vllm_bi_b1_r3` (left out, ref b=1 has 4 reps) and `prof0/repo3` (rerun once afterwards). The guard's load limit was relaxed to 150 because the load never fell below 14. Logs: /data41/audreymao/perf2_20261103/merge/out/.

## Bench (128 in / 128 out, tok/s, median [min-max], 5 reps, alternating base/merge per rep)

| T | B | base | merge | ratio | base spread | merge spread |
|---|---|---|---|---|---|---|
| 0 | 1 | 380 [306-382] | 446 [439-448] | 1.17 | 20% | 1.9% |
| 0 | 32 | 9,469 [8,513-9,628] | 10,988 [10,625-11,137] | 1.16 | 11.8% | 4.7% |
| 0 | 128 | 23,505 [23,293-23,646] | 27,176 [24,505-27,947] | 1.16 | 1.5% | 12.7% |
| 1 | 1 | 375 [345-377] | 441 [434-444] | 1.18 | 8.6% | 2.4% |
| 1 | 32 | 8,885 [8,841-9,016] | 9,398 [8,424-9,922] | 1.06 | 2.0% | 15.9% |
| 1 | 128 | 17,521 [15,332-19,151] | 18,024 [12,722-19,150] | 1.03 | 21.8% | 35.7% |

Spread = (max-min)/median. It exceeds 5% in 8 of 12 series. Cause: other users' jobs on the shared host (load 14-147 on 192 cores, plus intermittent processes on GPU 0). Frequency scaling is ruled out (governor performance). The T=1 steps are host bound (per-row PCG seeding in Python), which makes them most sensitive to host load. Min and max of each cell overlap only for T=1 B=32/128 and T=0 B=128 low end; T=0 B=1/32 and T=1 B=1 separate cleanly.

vLLM bf16 `VLLM_BATCH_INVARIANT=1` + `TRITON_ATTN` (`bench_attn.py`, same GPU, same session, T=0, 5 reps interleaved after the plugin bench; B=1 has 4): B=1 212 [210-214], B=32 7,418 [4,215-7,517] (one slow rep, 4 of 5 within 1.3%), B=128 32,297 [31,798-33,677]. `bench_attn.py` is copied from bitexact-llm/experiments/perf/det_compare (it is not present in the b128-merge worktree) and run with the same venv on 141.

## Profile, B=128 decode, ms/step (median of 3 runs per tree, interleaved)

| bucket | T=0 base | T=0 merge | delta | T=1 base | T=1 merge | delta |
|---|---|---|---|---|---|---|
| tok/s (profiler run) | 21,182 | 27,349 | | 18,872 | 19,392 | |
| wall | 4.860 | 3.917 | -0.94 | 6.160 | 5.935 | -0.23 |
| kernel_sum | 4.582 | 3.478 | -1.10 | 5.363 | 3.853 | -1.51 |
| attention | 1.259 | 0.696 | -0.56 | 1.258 | 0.696 | -0.56 |
| elementwise | 1.226 | 1.116 | -0.11 | 1.227 | 1.117 | -0.11 |
| sampling_logits | 0.579 | 0.405 | -0.17 | 1.357 | 0.757 | -0.60 |
| gemm_int8 + kv_write + other * | 1.516 | 1.257 | -0.26 | 1.516 | 1.256 | -0.26 |
| wall - busy (GPU idle) | 0.279 | 0.476 | +0.20 | 0.784 | 2.127 | +1.34 |

\* The bucketer does not know the new Triton int8 GEMM (`_k`, 1.347 ms over 64 calls) or the plane-store kernel, so they land in `other`; the three buckets are summed for a like-for-like comparison. The GEMM lane's kernel time is therefore 1.076 (cutlass) -> about 1.35 (Triton `_k` for all 64 GEMM calls, includes the shapes where it replaced cutlass); its gain in the merge is not visible in this bucket sum and needs the per-lane RESULTS (p2_gemm) for the isolated figure. The remaining kernels in `other` are `_h_kernel` 0.24 and `_quant_kernel` 0.07, unchanged.

Reading: GPU kernel time falls 1.1 ms (T=0) and 1.5 ms (T=1); at T=0 wall follows (-0.94 ms). At T=1 the step is host bound: GPU idle time grows to 2.1 ms, so the saved GPU time shows up only as -0.23 ms wall. The next lever for T>0 is host-side seeding, as noted in p2_sample.

## Not verified

- Per-lane contribution inside the merge (attn vs gemm vs sample): not run.
- A truly quiet-host bench: not obtainable on 141 today; T=1 B=32/128 and T=0 B=128 ratios are indicative only.
- The 1-run prof deltas for `wall - busy` are noisy for the same reason.

## Reproduce

`bash experiments/perf/p2_merge/run.sh sync`, then `STAGES="pytest suite tokens longctx bench ref prof" GPU=0 bash experiments/perf/p2_merge/run.sh start` (copy of the p2_attn runner with a load check, 5-rep bench and the vLLM reference stage).
