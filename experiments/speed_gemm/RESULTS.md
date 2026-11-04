# Hopper int8 GEMMs for the linear layers (branch exp/speed-gemm)

Result: calling cuBLASLt directly with a 32 MiB workspace gets sm90 wgmma int8 kernels that run the prefill-sized linear GEMMs 1.5-2.0x faster than `torch._int_mm`, with identical int32 outputs. With `BITEXACT_GEMM=lt` the scorer speeds up 1.06-1.09x and decode by 1-2.5%. Every gate passes. The default is unchanged (`torch._int_mm`).

Setup: one H100 80GB HBM3 per run (host 138, each GPU showed 0 MiB at start), torch 2.13.0+cu130 (cuBLASLt 13.1), Triton 3.7.1, vLLM 0.32.0. Llama-3.2-1B shapes. M is the row count passed to the GEMM, which is 2 activation limbs x tokens: M = 24 is decode B=1 (padded), 128 is decode B=64, 4096 is a 2,048-token prefill chunk.

## Why `torch._int_mm` runs an sm80 kernel

`torch._int_mm` calls `cublasLtMatmul` without a workspace. Under that constraint the cuBLASLt heuristic picks `cutlass_80_tensorop_i16832gemm_s8_*` (mma.sync). With a 32 MiB workspace on the same TN int8 -> int32 problem, the heuristic returns `sm90_xmma_gemm_i8i32_i8i32_i32_tn_n_*` wgmma kernels, some of them split-K / stream-K (a memset precedes them). Measured by enumerating up to 32 heuristic candidates per shape (`bench_lt_algos.py`, `results/lt.json`). Every candidate returned output equal to `torch._int_mm`.

## Options compared (`results/bench.json`, `results/path.json`)

- `torch._scaled_mm` with int8 inputs is not supported in torch 2.13: "Invalid scaling configuration", float8 only.
- Triton `tl.dot` int8 -> int32 (`bench_triton.py`, autotuned over 64-128 x 64-256 x 64-256 tiles, 3-5 stages, 4/8 warps): all 152 compiled variants contain `wgmma.mma_async` in their PTX, so Triton 3.7.1 does emit wgmma for int8 on sm90. It reaches 850-1,190 TOPS on prefill shapes, a 1.2-1.5x speed-up.
- cuBLASLt with a workspace (`bitexact.bmm.int_mm(a, w, "lt")`; on the first call for each shape it times the heuristic's 8 candidates against `torch._int_mm` and keeps the fastest): 960-1,390 TOPS on prefill shapes (48-70% of the 1,979 dense int8 peak), a 1.5-2.0x speed-up. Decode shapes are bound by weight bandwidth and change by 1.0-1.36x.

Bit equality: every Triton and cuBLASLt output in these tables equals `torch._int_mm` (`torch.equal` on int32, random int8 plus an all -128 row and column).

| layer (N x K) | M | `_int_mm` us / TOPS | Triton us / TOPS | cuBLASLt us / TOPS | Triton speed-up | cuBLASLt speed-up |
|---|---|---|---|---|---|---|
| qkv (3072 x 2048) | 24 | 11.6 / 26 | 11.6 / 26 | 13.7 / 22 | 1.05x | 0.85x |
| qkv (3072 x 2048) | 128 | 11.9 / 136 | 11.3 / 142 | 11.9 / 135 | 1.05x | 0.99x |
| qkv (3072 x 2048) | 1024 | 25.5 / 504 | 20.4 / 633 | 21.3 / 604 | 1.24x | 1.20x |
| qkv (3072 x 2048) | 4096 | 82.2 / 627 | 58.5 / 881 | 45.4 / 1134 | 1.42x | 1.81x |
| qkv (3072 x 2048) | 16384 | 313.3 / 658 | 216.4 / 953 | 161.6 / 1276 | 1.42x | 1.94x |
| o (2048 x 2048) | 24 | 11.6 / 17 | 11.2 / 18 | 11.3 / 18 | 1.03x | 1.03x |
| o (2048 x 2048) | 128 | 11.7 / 92 | 11.0 / 98 | 11.7 / 92 | 1.03x | 1.00x |
| o (2048 x 2048) | 1024 | 19.8 / 434 | 17.0 / 506 | 20.1 / 427 | 1.14x | 0.99x |
| o (2048 x 2048) | 4096 | 56.6 / 607 | 40.3 / 853 | 35.8 / 959 | 1.39x | 1.58x |
| o (2048 x 2048) | 16384 | 209.1 / 657 | 145.7 / 943 | 109.0 / 1261 | 1.42x | 1.92x |
| gate_up (16384 x 2048) | 24 | 22.8 / 71 | 21.9 / 73 | 22.5 / 71 | 1.03x | 1.01x |
| gate_up (16384 x 2048) | 128 | 26.7 / 321 | 23.8 / 361 | 24.8 / 347 | 1.12x | 1.08x |
| gate_up (16384 x 2048) | 1024 | 108.4 / 634 | 77.6 / 886 | 64.8 / 1060 | 1.40x | 1.67x |
| gate_up (16384 x 2048) | 4096 | 424.1 / 648 | 295.0 / 932 | 215.0 / 1278 | 1.39x | 1.97x |
| gate_up (16384 x 2048) | 16384 | 1801.4 / 610 | 1162.7 / 946 | 1008.2 / 1091 | 1.51x | 1.79x |
| down (2048 x 8192) | 24 | 24.2 / 33 | 22.5 / 36 | 17.8 / 45 | 1.17x | 1.36x |
| down (2048 x 8192) | 128 | 24.2 / 177 | 21.7 / 198 | 17.9 / 240 | 1.10x | 1.35x |
| down (2048 x 8192) | 1024 | 44.1 / 779 | 40.3 / 853 | 35.8 / 959 | 1.08x | 1.23x |
| down (2048 x 8192) | 4096 | 151.5 / 907 | 115.7 / 1188 | 99.5 / 1381 | 1.31x | 1.52x |
| down (2048 x 8192) | 16384 | 616.4 / 892 | 489.1 / 1124 | 395.6 / 1390 | 1.24x | 1.56x |
| lm_head (128256 x 2048) | 24 | 109.0 / 116 | 104.6 / 121 | 108.6 / 116 | 1.05x | 1.00x |
| lm_head (128256 x 2048) | 128 | 134.8 / 499 | 129.1 / 521 | 124.5 / 540 | 1.04x | 1.08x |
| lm_head (128256 x 2048) | 1024 | 851.9 / 631 | 569.6 / 944 | 441.8 / 1218 | 1.46x | 1.93x |
| lm_head (128256 x 2048) | 4096 | 3583.2 / 601 | 2417.7 / 890 | 1780.6 / 1208 | 1.45x | 2.01x |

Triton was also measured at M = 64, 512, 2048 and 8192 (`results/bench.json`), with the same pattern. Weighted over the four linear GEMMs of one layer, cuBLASLt is 1.80x at M = 4096, 1.39x at 1,024, 1.12x at 128 and 1.07x at 24. cuBLASLt beats Triton on every prefill shape, so cuBLASLt was wired in.

Exactness bound: the int8 operands are in [-128, 127], so the int32 sum over any subset of k is at most K x 2^14. For the largest K (8,192, down_proj) that is 2^27, below 2^31. int32 addition is exact, and alpha = 1, beta = 0 with an int32 scale type keeps the epilogue integer, so the summation order and split-K choice cannot change the result. `int_mm` asserts K < 2^17. `tests/test_lt_gemm.py` also checks a dot product of 127s with weights of 100-127, whose exact value (above 2^24 for K >= 1024) a float32 epilogue would round.

## Implementation

- `src/bitexact/bmm.py`: `int_mm(a, w)` plus the custom op `bitexact::lt_int_mm`, written with ctypes against the `libcublasLt` that torch already loads. It caches one plan per (M, N, K, row stride, alignments) and keeps one 32 MiB workspace per device. The candidates are timed on the first call for each shape, never during CUDA graph capture (until then it uses the heuristic's first choice). `BITEXACT_GEMM=lt` turns it on; the default is `intmm`, which leaves behavior unchanged. It works under `torch.compile(fullgraph=True)` and inside a CUDA graph (tested).
- `src/bitexact/int8/model.py`: `IntLlama._gemm` calls `int_mm(a, w)`, which covers every linear layer and the scorer's lm_head. The plugin's decode lm_head (`vllm_plugin/logits.py`) still calls `torch._int_mm`, because it is outside the write-set.
- `tests/test_lt_gemm.py`: equality on 6 shapes (including a row-strided `a`), plus the compiled and CUDA-graph paths.

## Gates (all with `BITEXACT_GEMM=lt`)

| gate | result |
|---|---|
| scorer, 8 configurations (GSM8K 128 seq at b8, b64, b1, b8 shuffled; 4k 16 seq at b4, b1, b8, b4 shuffled): rows equal plugin hash, sequence hash and float64 log p bits equal the eager reference | 8 / 8 (28,083 / 28,083 and 65,536 / 65,536 rows; 128 / 128 and 16 / 16 seqs) |
| decode `gate.py base gemm`: seeded T = 1 (53,387 tokens) and greedy (65,536 tokens), B=1 and B=64, against the pre-change plugin rollouts | 6 / 6 PASS, 0 sequences differ |
| full test suite (138, H100) | 90 passed, 5 skipped |
| `ruff check src tests experiments/speed*` | clean |

## Speed (same code, `BITEXACT_GEMM=intmm` vs `lt`, run at the same time on two GPUs)

| metric | `intmm` | `lt` | speed-up |
|---|---|---|---|
| scorer GSM8K b8, tok/s | 37,843 | 41,115 | 1.09x |
| scorer 4k b1, tok/s | 20,382 | 21,611 | 1.06x |
| scorer 4k b4, tok/s | 21,529 | 23,112 | 1.07x |
| decode B=1, ms/step | 2.797 | 2.728 | 1.03x |
| decode B=64, ms/step | 6.023 | 5.967 | 1.01x |
| prefill 64 requests, s | 0.576 | 0.569 | 1.01x |

Estimate vs measurement: the profile put GEMMs at 17.5% (GSM8K) and 8.5% (4k b4) of the compiled scorer at 27.5k / 12.9k tok/s. The current branch is faster everywhere else, so the same GEMM time is about 24% / 14% of today's wall. A 1.8x GEMM speed-up then predicts 1.12x / 1.07x. Measured: 1.09x / 1.07x. GSM8K falls short of the prediction because its lm_head (M = 512) and partial chunks run at smaller M, where the speed-up is lower. Decode GEMMs are bound by weight bandwidth, and the decode lm_head is not routed through this path, so the decode gain is small.

## Attention limb dots (tri_paged), estimate and tile sweep

- The fused attention kernels already use wgmma. In the Triton cache, `_max_kernel` / `_acc_kernel` / `_bmm_kernel` compiled with 64-row query tiles (scorer, prefill) contain `wgmma.mma_async`. The 16-row decode tiles use `mma.sync`, because wgmma needs at least 64 rows per warpgroup.
- Tile sweep of `tri_paged.attend` over bm in {16, 32, 64, 128}, bn in {32, 64, 128}, warps in {4, 8} (`attn_tiles.py`, `results/attn.json`). The best configuration against the default was 1.03x for the scorer at 4k b4 (1,024 query rows, 4,096 keys), 1.07x for GSM8K b8, 1.01x for decode B=64 at 2,304 keys, and 1.11x for decode B=64 at 512 keys. In the last case the best configuration is the default itself (bm 16, bn 64, 4 warps), so that 1.11x is timing noise. All outputs were equal.
- Conclusion: tile configs or wgmma cannot gain much here. Estimate, not profiled at instruction level here: the attention kernels are not bound by tensor-core throughput. Each 9-dot block sits among limb splits and int64 elementwise work (exp-table lookup, shifts, int64 combining), and tile shape barely moves the time. To make them faster, the per-element integer work has to shrink, or the limb count (which changes the numeric spec). A faster dot instruction does not help.

## Files

`bench_triton.py`, `bench_lt_algos.py`, `bench_path.py`, `attn_tiles.py`. Results are in `results/`: `bench.json` (Triton vs `_int_mm`), `lt.json` (best cuBLASLt candidate and kernel name per shape), `path.json` (`int_mm` lt path), `attn.json`, `scorer_gate.jsonl`, `decode_gate.txt`, `speed.jsonl`, `pytest.log`. The gate and speed runs used `experiments/speed/run_scorer.sh` and `run_decode.sh` with code tag `gemm` and `BITEXACT_GEMM` set in the environment.
