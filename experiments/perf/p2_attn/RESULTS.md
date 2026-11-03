# p2_attn: int8 limb-plane KV cache for decode attention

## Conclusion

1. **Bit-identity: PASS.** Full plugin suite (`tests/vllm/run_tests.sh`, 14 runs) for this branch (50be770) against base 971a3e1 on the same GPU: every run token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0; compare_exit 0 and compare_long_exit 0; all 14 hash files byte-identical (`cmp OK`). GPU pytest: 20 passed, 1 skipped. Bound stress (`extreme.py`, 43,680 keys, q/K/V at +-LIM[3], 3 shapes, nsplit auto/1/3/64, decode and general kernels): `EXTREME_OK`, equal to the CPU int64 reference. `attend()` on random caches equals base (`micro_new.py`).
2. **Decode attention kernel: 1.254 -> 0.694 ms/step at B=128** (profiler, `_dec_acc_kernel`, 16 layers), -0.56 ms (-45%). Kernel microbenchmark R=128 ctx=192: 74.6 -> about 43 us per layer call.
3. **Long context (S=16k) does not regress:** decode ms/step B=1 2.871 -> 2.554, B=8 7.699 -> 6.781.
4. **End-to-end tok/s is not established.** The GPU was noisy during the bench (rep 3 of several cells is 20-30% below reps 1-2), so the B=1/32/128 medians below do not separate base from branch. A quiet-machine bench is still to be run.

## What changed

- The cache row (512 bytes per key per KV head, vLLM-allocated) now holds the 3 signed base-256 int8 limbs of K and of V as planes: K limbs at bytes [0,192), V limbs at [192,384). Before: int32 K|V, re-split into limbs in both sweeps. 25% fewer bytes read and no in-kernel splitting.
- `tri_paged.store` / `_store_kernel` writes the planes (slot < 0 skipped); `paged.kv_view` returns an int8 view; `to_limbs` converts an int32 cache for tests.
- Decode path tile BN = 64 when R >= 64 (less masked work at short contexts), else 128.
- Integer semantics: the limbs are the same `_split` digits as before, only precomputed; q.K, p.V, max, table and division are unchanged.

## Numbers

Microbenchmark `attend()` (us per call, GPU 2): R=128 ctx=192 74.6 -> 43-51; R=128 ctx=256 80.8 -> 53-57; R=32 ctx=192 41.9 -> 42.3; R=8 ctx=16384 307 -> 256; R=1 ctx=16384 68.1 -> 67.3.

Bench T=0, 128 in / 128 out, tok/s, 3 reps (base | branch), noisy GPU:

| B | base | branch | median base | median branch |
|---|---|---|---|---|
| 1 | 378 / 408 / 305 | 412 / 393 / 281 | 378 | 393 |
| 32 | 9,095 / 9,389 / 6,125 | 7,851 / 9,556 / 8,613 | 9,095 | 8,613 |
| 128 | 17,908 / 21,103 / 15,292 | 17,804 / 20,802 / 26,004 | 17,908 | 20,802 |

Profiler B=128 (one run each, same noisy session, so wall is higher than the 5.3 ms of earlier sessions): kernel_sum 4.588 -> 3.889 ms, attention (`_dec_acc_kernel`) 1.257 -> 0.696, elementwise 1.229 -> 1.130, gemm_int8 1.075 -> 1.078.

## What did not work

A BN / num_warps sweep on the old int32 layout gained at most about 20% (R=128 ctx=192: 59 us at bn=32 w=4 or bn=64 w=2 versus 75 us); BN=128 with 1 warp was 5x slower. GQA grouping was already in place (4 query heads per KV head in one CTA, 1024 CTAs at B=128), so it was not a lever.

## Not verified

- End-to-end tok/s on a quiet GPU (above).
- The extra cost of `_store_kernel` in the engine step (the profiler shows kernel time fell overall, so it is at most small).
- Narrowing below 3 bytes per element: the value bounds (LIM[3] about 2^23) need 3 limbs, so this is the floor without changing semantics.
- Outputs on 141: /data41/audreymao/perf2_20261103/attn/out/.
