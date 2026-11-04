# p3_splitkv: split-KV rule for integer decode attention

## Conclusion
1. Bit-identity PASS. Full 14-run suite, branch vs base 07aa43c, same GPU: all token_seqs_diff 0, logit_hash_diff 0, missing 0, compare_exit 0, compare_long_exit 0; 14/14 hash files `cmp OK`. GPU pytest 27 passed, 1 skipped. extreme.py (key bound, nsplit auto/1/3/64): EXTREME_OK.
2. No new kernel was needed. The decode path already had split-KV (`_dec_max_kernel` global max by atomic max, then `_dec_acc_kernel` per-split num/den by int64 atomic add; sums are exact in any order). The change is the static nsplit rule and bn:
   - bn = 64 at every R (was 128 for R < 64).
   - R <= 8: nsplit = min(64, 528 // (R*8)) (was 264 // tiles).
   - R > 8: 1 if 2*tiles >= 264 (as before); otherwise as above, capped at table_keys // 2048.
3. Long context decode ms/step (longctx_step, CUDA graph, GPU 3):

| S | B | base | branch |
|---|---|---|---|
| 4k | 1 | 1.940 | 1.925 |
| 4k | 8 | 3.326 | 3.121 |
| 16k | 1 | 2.377 | 2.274 |
| 16k | 8 | 6.582 | 5.877 |
| 32k | 1 | 2.961 | 2.753 |
| 32k | 8 | 10.895 | 9.519 |

   Gains are modest: B=8 -11% at 16k, -13% at 32k; B=1 -4 to -7%. Base B=1 already split 33 ways, so the problem statement "8 CTAs at B=1" does not hold on this base; the remaining B=1 time is not attention-split bound.
4. B=128 short context: attention `_dec_acc_kernel` 0.693 -> 0.694 ms (unchanged: R=128 gets nsplit 1 as before). Profiler decode_step_ms 3.818 (base) vs 3.874 (branch), kernel_sum 3.42 vs 3.47: run-to-run noise, no real change. Bench tok/s medians of 5 (T=0, max_model_len 2048): B=1 450.7 vs 455.1, B=32 11,082 vs 11,133, B=128 29,141 vs 29,214. All within noise; B=128 gain < 3%, so this lane is not a B=128 win.

## Micro (attend(), us per layer call, noisy shared GPU)
R=8 ctx=32k 1075 -> 458; R=8 16k 283 -> 242; R=16 16k 537 -> 457; R=1 192 and R=8 192 unchanged or faster; R>=32 short ctx equal; R=128 equal.

## Didn't work
- Key cap of 1024 per split at all R: B=1 S=4k 1.94 -> 2.83 and B=1 16k 2.38 -> 2.94 ms (base uses 33 splits there), and B=32 -3.5% tok/s. Cap restricted to R > 8 with 2048 keys.
- A split costs about 30 us (3 launches + zero fills); at ctx <= 1k one fused CTA (37 us) beats any split (68 us). The block table is as wide as max_model_len, not the real context, so a static rule cannot see short real contexts; the R > 8 cap keeps base behavior there.
- bn/warps sweep: bn 64, warps 4 best nearly everywhere.

## Not verified
- Quiet-machine tok/s (GPU 3 shared with lane lmhead; first run of the earlier rule had the B=32 regression, final run is inside noise).
- Longctx for B between 2 and 7 and 9 to 32 beyond micro numbers.
- Outputs on 141: /data41/audreymao/perf3_20261103/splitkv/out.
