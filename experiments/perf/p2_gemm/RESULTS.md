# perf2 / gemm: skinny-M Triton int8 GEMM for decode

Conclusion: bit-identity PASS; speedup is small. B=128 +2.8% tok/s (median of 3, below the 3% bar; kernel time -0.18 ms/step, about 3.4% of the 5.3 ms step); B=32 +13%; B=1 unchanged. Left unmerged (branch `perf2/gemm`) unless B=32 counts.

## What changed
`src/bitexact/int8/gemm_skinny.py`: Triton int8 GEMM (small BLOCK_M, large BLOCK_K, optional split-K with int32 atomic add) and `int_mm()` dispatch used by `IntLlama._gemm`. Static table of configs per (N, K) and row bucket (<=24, <=64, <=256), chosen offline by `search_graph.py` (CUDA-graph timing, HBM-cold rotating weights). Serve time uses no timing, so it is deterministic; integer sums are exact, so any config gives the same integers (K*127*128 < 2^31). qkv, o, gate_up, down use Triton; LM head and rows > 256 stay on torch._int_mm (Triton was slower for the LM head: 126 us vs 97 us at 24 rows, 243 vs 192 at 64).

## Microbench (H800 GPU 1, CUDA graph, cold weights, us; roofline = weight bytes / 3.35 TB/s)
| shape (N,K) | rows | _int_mm | Triton | roofline |
|---|---|---|---|---|
| qkv 3072x2048 | 24 / 64 / 256 | 7.5 / 7.6 / 9.3 | 5.8 / 6.7 / 8.4 | 1.9 |
| o 2048x2048 | 24 / 64 / 256 | 7.3 / 7.4 / 7.5 | 5.0 / 5.4 / 6.4 | 1.3 |
| gate_up 16384x2048 | 24 / 64 / 256 | 15.3 / 18.7 / 42.1 | 15.0 / 18.2 / 30.4 | 10.0 |
| down 2048x8192 | 24 / 64 / 256 | 26.8 / 19.6 / 19.8 | 10.3 / 12.5 / 15.6 | 5.0 |
| LM head 128256x2048 | 24 / 64 | 97 / 192 | 126 / 243 | 78 |

Small shapes sit at launch/latency floor (~5 us), so the roofline is not reachable there; gate_up is already near 1.5x roofline. Only down has large headroom, and it is taken.

## Bit-identity
pytest (20 passed, 1 skipped, incl. new tests/test_gemm_skinny.py: Triton == _int_mm on all table shapes, extreme values -128/127). smoke: base_ref vs repo alone_vllmgraph, token_seqs_diff 0, logit hashes `cmp OK`. The 14-job suite was not run.

## tok/s, T=0, 128 in / 128 out, 3 interleaved runs, median (min..max)
| B | base 971a3e1 | branch | delta |
|---|---|---|---|
| 1 | 353 (1 run) | 363 (1 run) | within noise |
| 32 | 7,785 (7,393..7,788) | 8,827 (8,260..9,093) | +13% |
| 128 | 20,235 (14,277..20,338) | 20,810 (16,844..23,324) | +2.8% |

Run-to-run noise is large (shared host); the profile below is the more reliable signal.

## Profile, B=128 decode, ms/step
gemm_int8 body (qkv/o/gu/down): 1.069 -> 0.891 (Triton `_k`, classified "other"); kernel_sum 4.606 -> 4.402. LM head unchanged (0.209). Profiled wall 7.74 -> 6.32 is mostly host idle noise.

## Not verified / not tried
LM head alternative (split-N Triton, cuBLASLt via other torch paths) not tried beyond the Triton sweep; 14-job suite, longctx, T=1 repeats not rerun; B=1 and T=1 single runs only.
