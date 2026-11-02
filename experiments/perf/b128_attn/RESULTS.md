# perf/attn (limb-stacked decode attention, 5601138): verification and throughput

## Conclusion

1. **Bit-identity: PASS.** The full plugin suite (`tests/vllm/run_tests.sh`, 14 runs) was run for 5601138 and for 29e5cb5 on the same GPU. Both trees: compare_exit 0 and compare_long_exit 0, every run token_seqs_diff 0, missing 0, logit_hash_diff 0. All 14 hash files of 5601138 are byte-identical (`cmp`) to the 29e5cb5 files of this session and to the d87b9ce files of the combo session (d87b9ce has the same `src/` and `tests/` as 29e5cb5). GPU pytest passes (10 passed; the 1 skip, `test_overflow_check.py`, passes when run with its model variable set).
2. **Integer semantics are unchanged.** The new kernels compute the same integers as the old ones by construction (argument below), and an extra stress test at the bound (43,680 keys, all values at ±LIM[3]) matches the CPU int64 reference and the old kernels exactly.
3. **Throughput at T=0** (bench, median of 3 alternating passes, same GPU and session): B=128 **17,523 → 19,956 tok/s (1.14x)**, B=32 8,176 → 8,885 (1.09x), B=1 370 → 369 (no change, within noise). Long context S=16k decode step: B=8 **11.74 → 7.84 ms (1.50x)**, B=1 3.16 → 2.95 ms (1.07x).
4. Decode attention kernel time per step (profiler) drops 1.67 → 1.30 ms at B=128 and 0.52 → 0.37 ms at B=32. At B=128 the plugin reaches about 0.57x of vLLM bf16 BATCH_INVARIANT + TRITON_ATTN (34,941 tok/s, combo session, not re-measured). Attention is still 25% of the B=128 step, and elementwise (1.77 ms) remains the largest bucket.

## What changed

`src/bitexact/vllm_plugin/tri_paged.py`: for M ≤ 4 query rows per (request, KV head), which is every decode step (G = 4 query heads per KV head), `attend` dispatches to new kernels `_dec_max_kernel` / `_dec_acc_kernel`:

- The 3 int8 limbs of the 4 queries are stacked as rows of one 16-row int8 operand (row i·4 + g = limb i of query g, rows 12–15 zero). q·K takes 3 tensor-core dots (one per K limb) instead of 9, and p·V takes 3 instead of 9.
- With one key split (chosen when R·HKV ≥ 132 programs, i.e. B ≥ 17) a single kernel takes the row maxima in a first sweep and the sums in a second sweep, then stores num / den directly (no max kernel launch, no atomics, no zero fills). Otherwise the split path keeps the two kernels with atomic max / add, as before.
- Default key tile BN = 128 for the decode path; general path unchanged (bm=16 forces it, which the test uses as a second reference).

`tests/test_vllm_kernels.py`: adds M = 1 and M = 3 cases and runs every case through both the decode and the general kernels.

## Integer-semantics argument (reviewed)

- Limbs: both paths use the same `_split` (signed limbs in [−128, 127]), applied to the same int32 values (q loaded as int32, K/V from the int32 cache, p from the int32 weight table). Hence the same limbs a_i, b_j.
- q·K: stacked dot entry (row i·4+g, key s) = Σ_d a_i[g,d]·b_j[s,d], |·| ≤ 64·2^14 = 2^20 < 2^31, exact in int32. `_unstack` forms Σ_ij c_ij·2^(8(i+j)) in int64. The old `_limb_dot` sums the same products grouped by i+j (each int32 group ≤ 3·2^20) and then shifts in int64. Both equal the exact integer q·K (|q·K| ≤ 64·LIM[3]^2 < 2^53), so `raw` is bit-identical.
- Max, z = rsh(raw − mx), table lookup, validity mask (s ≤ qpos, s < hi): same expressions on the same `raw`; the max is over the same set of keys (fused mode: lo = 0, hi = kend, one program sees all keys). Integer max is order-independent.
- p·V: e_j[i·4+g, d] = Σ_s p_i[g,s]·v_j[s,d] accumulated in int32 over all keys of the program. Per key |p_i·v_j| ≤ 2^14, and keys < 43,690 (asserted in `attend`), so |e_j| < 43,690·2^14 ≈ 7.2·10^8 < 2^31: no wrap. The old kernel's per-group accumulators had the tighter-to-fit bound 3·2^14·43,690 < 2^31; the new bound is 3x looser. num is formed in int64 as Σ_ij e·2^(8(i+j)) and equals the exact Σ_s p·V (values < 2^63 since p ≤ 2^20 and |V| ≤ LIM[3]). den = Σ p in int64, as before.
- Split path: per-split int64 partial sums combined by atomic add, maxima by atomic max, exactly as the old kernels (integer add is associative, no intermediate bound changes).
- Padded rows (g ≥ M) have q = 0, qpos = −1, so no valid key; their outputs are masked on store.
- Rounding and saturation points (rsh of the score difference, table clamp, final num / den division in `attn_div`) are untouched.

Stress test at the bound (`experiments/perf/b128_attn/extreme.py`, GPU 1): 2,730 blocks × 16 = 43,680 keys, q, K, V all = −LIM[3], all = +LIM[3], random ±LIM[3], and K = −LIM[3] with V = ±LIM[3]; (R, M) ∈ {(1,4), (2,1), (1,3)}; nsplit ∈ {auto, 1, 3, 64}, decode and general kernels. All equal to the CPU int64 reference (`segs.softmax` + einsum); max |num| 3.8·10^17, max den 4.6·10^10. `EXTREME_OK`.

## Bit-identity

Server 141, GPU 7, 02:30–02:52 server clock, `experiments/perf/b128_attn/verify.sh`, outputs `/data41/audreymao/b128_20261031/attn/verify/out/`. Trees exported with `git archive` (5601138 → `verify/repo`, 29e5cb5 → `verify/base`).

| check | result |
|---|---|
| suite 5601138: 10 short runs vs `run_ref` (12 targets × 48 positions, T = 0 / 0.7 / 1.0), 4 long runs vs `long_ref` (3 × 16) | all token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0; compare_exit 0, compare_long_exit 0 |
| suite 29e5cb5, same GPU | same, after one rerun (below) |
| `cmp repo_X.hash base_X.hash`, 14 X | 14/14 byte-identical |
| `cmp repo_X.hash` vs combo-session d87b9ce `repo_X.hash`, 14 X | 14/14 byte-identical |
| GPU pytest (`-m gpu tests`, 5601138) | 10 passed, 1 skipped; the skipped `test_overflow_check.py` passes with `BITEXACT_VLLM_MODEL` set |
| decode-kernel stress at the key bound | pass (above) |

The baseline run `base_alone_pc0` first failed in vLLM's start-up memory check (`Error in memory profiling`: another job of this user on GPU 7 changed free memory during start-up). It was rerun alone on GPU 7 (`rerun.sh`): diffs 0, and its hash is byte-identical to the 5601138 file and the combo file.

## Throughput

`experiments/perf/b128_attn/bench2.sh`, GPU 1, 03:19–03:38, outputs `verify/out2/`. bench.py: 128 random input tokens, 128 output tokens, ignore_eos, CUDA graph, max_num_seqs = B, second of two `generate` calls timed. base and repo alternate per cell, 3 passes. The script polled the GPU every 2 s during each run; no other process appeared on GPU 1 (max_gpu_procs = 1 for every run).

### bench, T=0, output tok/s

| B | 29e5cb5 (3 passes) | 5601138 (3 passes) | median base | median repo | speedup |
|---|---|---|---|---|---|
| 1 | 372.1 / 365.7 / 369.9 | 386.7 / 356.5 / 368.9 | 369.9 | 368.9 | 1.00x |
| 32 | 8,808.6 / 8,143.1 / 8,175.6 | 8,943.2 / 8,885.0 / 8,359.2 | 8,175.6 | 8,885.0 | 1.09x |
| 128 | 17,523.0 / 17,120.2 / 18,204.4 | 19,298.0 / 19,956.2 / 20,327.5 | 17,523.0 | 19,956.2 | 1.14x |

Values are out_tokens / sec from each RESULT line (e.g. 16,384 / 0.821 s = 19,956). This session's baseline at B=128 is about 9% below the 19,354 of the combo session (GPU 0); the comparison here is same GPU, same session.

### long context, `tests/vllm/longctx_step.py --S 16384 --prof`, decode ms per step

| B | 29e5cb5 (2 passes) | 5601138 (2 passes) | ratio |
|---|---|---|---|
| 1 | 3.167 / 3.158 | 2.950 / 2.948 | 1.07x |
| 8 | 11.734 / 11.742 | 7.843 / 7.838 | 1.50x |

### profiler, decode ms per step (`prof_vllm.py` + `buckets.py`, GPU 1)

| run | wall | kernel busy | attention | elementwise | gemm_int8 | sampling+logits |
|---|---|---|---|---|---|---|
| 29e5cb5 B=128 | 5.95 | 5.74 | 1.67 (`_acc_kernel` 1.18 + `_max_kernel` 0.49) | 1.88 | 1.15 | 0.91 |
| 5601138 B=128 | 5.38 | 5.22 | 1.30 (`_dec_acc_kernel` only) | 1.77 | 1.12 | 0.91 |
| 29e5cb5 B=32 | 3.60 | 3.26 | 0.52 | 1.40 | 0.81 | 0.41 |
| 5601138 B=32 | 3.29 | 3.01 | 0.37 | 1.34 | 0.82 | 0.37 |

## Caveats

- **Not verified:** T=1 throughput on a clean GPU. A T=1 pass ran in `verify.sh` on GPU 7 while the elt lane shared that GPU (several bench runs there failed at start-up or were slowed), so those numbers are not used. T=1 bit-identity is covered by the suite (T = 0.7 / 1.0 targets).
- The production path without the hash log was not re-checked with `tok_digest.py`; the bench runs it, but only throughput was recorded.
- The bf16 BI reference (34,941 tok/s) is from the combo session on GPU 0, not re-measured here.
- The B=1 cells vary by ±4% between passes; no B=1 change is claimed in either direction.
- The decode path (M ≤ 4) relies on the existing `attend` assert of < 43,690 keys per request for the p·V int32 bound; that assert is unchanged.
