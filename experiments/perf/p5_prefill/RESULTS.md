# p5_prefill: prefill throughput, 1B (H800, GPU 7 benches, base a0354e1 vs perf5/prefill)

Change: prefill / mixed steps run tri_fuse's kernels (4 per layer instead of ~28 Inductor/ATen kernels), a token-major
prefill attention kernel (`tri_prefill.py`: both sweeps in one program, no gather/scatter/atomics), and Triton int8 GEMM
tiles for M >= 2048 rows (`gemm_skinny.LARGE`, 1.2-1.4x over cutlass `_int_mm`). Same integers.

## Prefill tok/s (prompt tokens/s, 1 step, median of 3 interleaved reps)
| B x L | base | new | ratio | vLLM stock | vLLM BI+TRITON_ATTN |
|---|---|---|---|---|---|
| 64 x 256 | 67,181 | 106,568 | 1.59 | 243,994 | 179,575 |
| 64 x 512 | 65,361 | 100,247 | 1.53 | 266,964 | 183,279 |
| 16 x 512 | 62,615 | 99,862 | 1.59 | 251,280 | 172,882 |
| 16 x 1024 | 58,680 | 85,110 | 1.45 | 258,806 | 170,896 |
| 32 x 1024 | 59,020 | 86,089 | 1.46 | 265,881 | 170,920 |
| 64 x 1024 | 58,966 | 82,925 | 1.41 | 261,653 | 171,591 |

## Profile, 64 x 256 prefill step (ms of GPU kernel time)
base 230.3 (853 kernels): int8 GEMM 83.3, elementwise 75.9, attention 39.1, other 28.3, kv_write 3.4.
new 141.1 (226 kernels): GEMM 71.1 (Triton `_k`) + tri_mlp `_h`/`_quant` 26.8, `_pre_kernel` attention 22.4,
fused elementwise 20.3. 64 x 512: 485.5 -> 310.2 ms (attention 106 -> 75).
Remaining floor: GEMMs (~50%), `_h_kernel` silu epilogue 20 ms (a fused GEMM+silu epilogue was tried and was 1.25x slower, dropped).

## End-to-end bench (128 in / 128 out, T=0, 5 reps, tok/s; includes the prefill)
B=1 531 -> 556, B=32 13,240 -> 14,157, B=128 29,530 -> 30,796. Decode-only step unchanged (64 x 256 profile:
2.60-2.66 ms both; 512: 2.91-2.95 both; longctx S=16k decode 1.971/5.21 -> 1.972/5.12 ms, 512-chunk prefill 73.9/73.5 -> 74.7/74.6 ms,
long-context chunk path is unchanged because MIN_TILES keeps few-tile steps on the old kernels).

## Bit identity
pytest 129 passed 2 skipped (new tests: test_tri_prefill, test_large_equals_int_mm); 1B 14-run suite: all 14 `.hash` cmp OK vs base,
token_seqs_diff 0 / logit_hash_diff 0; token dumps (mixed temps, T=0, logprobs) IDENTICAL vs base; 8B plugin vs IntLlama reference
(s8 7 runs) compare_exit 0, 8B batch_inv PASS (graph + eager).
