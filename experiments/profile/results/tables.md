## Scorer

### sc_4k_b1_comp: q2_4096_B.json, batch 1, compiled, nl 2

unprofiled: 8935.8 tok/s over 16 batches (68947 tokens, sec [8.393, 7.038]); padding: {'real_tokens': 68947, 'computed_tokens': 68947, 'token_util': 1.0, 'real_qk_pairs': 148590962, 'computed_qk_pairs': 157311377, 'qk_util': 0.9446}
profiled: 1 batches, 4311 tokens, wall 0.916 s, 15383 kernels (3.6 per token)

**sc_4k_b1_comp: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 32.04 | 1088 | 8.4 | 3.5 |
| int GEMM, lm_head | 6.14 | 16 | 1.6 | 0.7 |
| attention QK (bmm) | 163.40 | 272 | 42.7 | 17.8 |
| attention PV (bmm) | 39.21 | 272 | 10.2 | 4.3 |
| attention softmax (mask, max, shift, sum) | 42.83 | 272 | 11.2 | 4.7 |
| attention num/den divide (compiled: + quantization) | 7.28 | 1104 | 1.9 | 0.8 |
| RMSNorm + quantization (compiled segment) | 10.85 | 1152 | 2.8 | 1.2 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 19.11 | 864 | 5.0 | 2.1 |
| GEMM epilogue (limb join, rescale, residual add) | 33.38 | 3312 | 8.7 | 3.6 |
| RoPE / q,k,v split | 8.21 | 2448 | 2.1 | 0.9 |
| other elementwise | 8.74 | 4503 | 2.3 | 1.0 |
| log-softmax float64 (scorer) | 11.84 | 80 | 3.1 | 1.3 |
| **sum of kernel time** | 383.02 | 15383 | 100 | 41.8 |
| GPU busy (union of kernels) | 383.02 | | | 41.8 |
| GPU idle: launch / Python / host sync | 532.98 | | | 58.2 |
| wall | 916.00 | | | 100 |

**sc_4k_b1_comp: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| attention (bmm q.k / p.v) | 202.61 | 544 | 52.9 | 22.1 |
| fused elementwise (Inductor) | 121.65 | 9152 | 31.8 | 13.3 |
| int GEMM (cuBLASLt _int_mm) | 38.18 | 1104 | 10.0 | 4.2 |
| eager elementwise | 14.41 | 4534 | 3.8 | 1.6 |
| other | 6.03 | 16 | 1.6 | 0.7 |
| eager reduction | 0.14 | 33 | 0.0 | 0.0 |
| GPU busy | 383.02 | | | 41.8 |
| GPU idle | 532.98 | | | 58.2 |

**sc_4k_b1_comp: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 512 | 16384 | 2048 |  | 256 | 52.93 | 649.1 | 33 |
| 512 | 2048 | 8192 |  | 256 | 39.35 | 436.6 | 22 |
| 512 | 128256 | 2048 | y | 16 | 383.83 | 700.8 | 35 |
| 512 | 3072 | 2048 |  | 256 | 15.3 | 421.0 | 21 |
| 512 | 2048 | 2048 |  | 256 | 10.75 | 399.7 | 20 |
| 432 | 16384 | 2048 |  | 16 | 45.95 | 630.9 | 32 |
| 432 | 2048 | 8192 |  | 16 | 39.24 | 369.4 | 19 |
| 432 | 3072 | 2048 |  | 16 | 14.88 | 365.4 | 18 |
| 432 | 2048 | 2048 |  | 16 | 8.93 | 405.7 | 21 |

### sc_4k_b1_eager: q2_4096_B.json, batch 1, eager, nl 2

unprofiled: 2415.7 tok/s over 16 batches (68947 tokens, sec [28.541, 28.541]); padding: {'real_tokens': 68947, 'computed_tokens': 68947, 'token_util': 1.0, 'real_qk_pairs': 148590962, 'computed_qk_pairs': 157311377, 'qk_util': 0.9446}
profiled: 1 batches, 4311 tokens, wall 3.814 s, 216695 kernels (50.3 per token)

**sc_4k_b1_eager: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 32.36 | 1088 | 2.5 | 0.8 |
| int GEMM, lm_head | 6.25 | 16 | 0.5 | 0.2 |
| attention QK (bmm) | 141.77 | 272 | 11.1 | 3.7 |
| attention PV (bmm) | 38.87 | 272 | 3.0 | 1.0 |
| attention softmax (mask, max, shift, sum) | 300.80 | 3808 | 23.6 | 7.9 |
| exp LUT | 159.80 | 1632 | 12.5 | 4.2 |
| attention num/den divide (compiled: + quantization) | 6.16 | 1904 | 0.5 | 0.2 |
| RMSNorm | 64.66 | 42000 | 5.1 | 1.7 |
| RMSNorm + quantization (compiled segment) | 8.05 | 5040 | 0.6 | 0.2 |
| quantization (quant_rows, limb split) | 203.97 | 98384 | 16.0 | 5.3 |
| SiLU LUT | 52.91 | 4896 | 4.1 | 1.4 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 19.02 | 2720 | 1.5 | 0.5 |
| GEMM epilogue (limb join, rescale, residual add) | 200.04 | 40288 | 15.7 | 5.2 |
| RoPE / q,k,v split | 21.13 | 9792 | 1.7 | 0.6 |
| other elementwise | 8.85 | 4503 | 0.7 | 0.2 |
| log-softmax float64 (scorer) | 12.05 | 80 | 0.9 | 0.3 |
| **sum of kernel time** | 1276.69 | 216695 | 100 | 33.5 |
| GPU busy (union of kernels) | 1276.69 | | | 33.5 |
| GPU idle: launch / Python / host sync | 2537.31 | | | 66.5 |
| wall | 3814.00 | | | 100 |

**sc_4k_b1_eager: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 987.92 | 209766 | 77.4 | 25.9 |
| attention (bmm q.k / p.v) | 180.64 | 544 | 14.1 | 4.7 |
| eager reduction | 63.44 | 5265 | 5.0 | 1.7 |
| int GEMM (cuBLASLt _int_mm) | 38.60 | 1104 | 3.0 | 1.0 |
| other | 6.09 | 16 | 0.5 | 0.2 |
| GPU busy | 1276.69 | | | 33.5 |
| GPU idle | 2537.31 | | | 66.5 |

**sc_4k_b1_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 512 | 16384 | 2048 |  | 256 | 53.32 | 644.3 | 33 |
| 512 | 2048 | 8192 |  | 256 | 39.63 | 433.5 | 22 |
| 512 | 128256 | 2048 | y | 16 | 390.59 | 688.6 | 35 |
| 512 | 3072 | 2048 |  | 256 | 15.38 | 418.9 | 21 |
| 512 | 2048 | 2048 |  | 256 | 11.09 | 387.3 | 20 |
| 432 | 16384 | 2048 |  | 16 | 47.35 | 612.2 | 31 |
| 432 | 2048 | 8192 |  | 16 | 39.67 | 365.4 | 18 |
| 432 | 3072 | 2048 |  | 16 | 15.17 | 358.3 | 18 |
| 432 | 2048 | 2048 |  | 16 | 9.21 | 393.5 | 20 |

### sc_4k_b4_comp: q2_4096_B.json, batch 4, compiled, nl 2

unprofiled: 12919.7 tok/s over 4 batches (68947 tokens, sec [5.342, 5.331]); padding: {'real_tokens': 68947, 'computed_tokens': 69416, 'token_util': 0.9932, 'real_qk_pairs': 148590962, 'computed_qk_pairs': 159440312, 'qk_util': 0.932}
profiled: 1 batches, 17164 tokens, wall 1.41 s, 17223 kernels (1.0 per token)

**sc_4k_b4_comp: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 93.83 | 1088 | 7.3 | 6.7 |
| int GEMM, lm_head | 24.98 | 64 | 1.9 | 1.8 |
| attention QK (bmm) | 550.57 | 272 | 42.7 | 39.0 |
| attention PV (bmm) | 127.11 | 272 | 9.9 | 9.0 |
| attention softmax (mask, max, shift, sum) | 164.95 | 272 | 12.8 | 11.7 |
| attention num/den divide (compiled: + quantization) | 23.47 | 1088 | 1.8 | 1.7 |
| RMSNorm + quantization (compiled segment) | 37.40 | 1216 | 2.9 | 2.7 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 87.98 | 1632 | 6.8 | 6.2 |
| GEMM epilogue (limb join, rescale, residual add) | 104.19 | 3456 | 8.1 | 7.4 |
| RoPE / q,k,v split | 14.50 | 2448 | 1.1 | 1.0 |
| other elementwise | 12.84 | 5095 | 1.0 | 0.9 |
| log-softmax float64 (scorer) | 47.46 | 320 | 3.7 | 3.4 |
| **sum of kernel time** | 1289.27 | 17223 | 100 | 91.4 |
| GPU busy (union of kernels) | 1289.27 | | | 91.4 |
| GPU idle: launch / Python / host sync | 120.73 | | | 8.6 |
| wall | 1410.00 | | | 100 |

**sc_4k_b4_comp: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| attention (bmm q.k / p.v) | 677.68 | 544 | 52.6 | 48.1 |
| fused elementwise (Inductor) | 432.48 | 10112 | 33.5 | 30.7 |
| int GEMM (cuBLASLt _int_mm) | 118.80 | 1152 | 9.2 | 8.4 |
| eager elementwise | 35.80 | 5270 | 2.8 | 2.5 |
| other | 24.22 | 64 | 1.9 | 1.7 |
| eager reduction | 0.28 | 81 | 0.0 | 0.0 |
| GPU busy | 1289.27 | | | 91.4 |
| GPU idle | 120.73 | | | 8.6 |

**sc_4k_b4_comp: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 2048 | 16384 | 2048 |  | 256 | 201.3 | 682.7 | 34 |
| 512 | 128256 | 2048 | y | 64 | 390.28 | 689.2 | 35 |
| 2048 | 2048 | 8192 |  | 256 | 78.98 | 870.1 | 44 |
| 2048 | 3072 | 2048 |  | 256 | 39.96 | 644.9 | 33 |
| 2048 | 2048 | 2048 |  | 256 | 27.46 | 625.5 | 32 |
| 1720 | 16384 | 2048 |  | 16 | 165.87 | 695.9 | 35 |
| 1720 | 2048 | 8192 |  | 16 | 76.01 | 759.3 | 38 |
| 1720 | 3072 | 2048 |  | 16 | 32.66 | 662.6 | 33 |
| 1720 | 2048 | 2048 |  | 16 | 26.36 | 547.5 | 28 |

### sc_4k_b4_eager: q2_4096_B.json, batch 4, eager, nl 2

unprofiled: 3637.2 tok/s over 4 batches (68947 tokens, sec [18.984, 18.928]); padding: {'real_tokens': 68947, 'computed_tokens': 69416, 'token_util': 0.9932, 'real_qk_pairs': 148590962, 'computed_qk_pairs': 159440312, 'qk_util': 0.932}
profiled: 1 batches, 17164 tokens, wall 5.117 s, 227479 kernels (13.3 per token)

**sc_4k_b4_eager: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 94.19 | 1088 | 2.2 | 1.8 |
| int GEMM, lm_head | 24.82 | 64 | 0.6 | 0.5 |
| attention QK (bmm) | 527.83 | 272 | 12.1 | 10.3 |
| attention PV (bmm) | 125.63 | 272 | 2.9 | 2.5 |
| attention softmax (mask, max, shift, sum) | 1178.69 | 3808 | 27.1 | 23.0 |
| exp LUT | 624.31 | 1632 | 14.4 | 12.2 |
| attention num/den divide (compiled: + quantization) | 18.28 | 1904 | 0.4 | 0.4 |
| RMSNorm | 88.19 | 45600 | 2.0 | 1.7 |
| RMSNorm + quantization (compiled segment) | 15.42 | 5472 | 0.4 | 0.3 |
| quantization (quant_rows, limb split) | 551.22 | 102528 | 12.7 | 10.8 |
| SiLU LUT | 248.64 | 4896 | 5.7 | 4.9 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 83.77 | 2720 | 1.9 | 1.6 |
| GEMM epilogue (limb join, rescale, residual add) | 663.54 | 42016 | 15.3 | 13.0 |
| RoPE / q,k,v split | 40.52 | 9792 | 0.9 | 0.8 |
| other elementwise | 12.87 | 5095 | 0.3 | 0.3 |
| log-softmax float64 (scorer) | 48.18 | 320 | 1.1 | 0.9 |
| **sum of kernel time** | 4346.09 | 227479 | 100 | 84.9 |
| GPU busy (union of kernels) | 4346.09 | | | 84.9 |
| GPU idle: launch / Python / host sync | 770.91 | | | 15.1 |
| wall | 5117.00 | | | 100 |

**sc_4k_b4_eager: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 3367.19 | 220214 | 77.5 | 65.8 |
| attention (bmm q.k / p.v) | 653.45 | 544 | 15.0 | 12.8 |
| eager reduction | 182.10 | 5505 | 4.2 | 3.6 |
| int GEMM (cuBLASLt _int_mm) | 119.01 | 1152 | 2.7 | 2.3 |
| other | 24.35 | 64 | 0.6 | 0.5 |
| GPU busy | 4346.09 | | | 84.9 |
| GPU idle | 770.91 | | | 15.1 |

**sc_4k_b4_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 2048 | 16384 | 2048 |  | 256 | 202.48 | 678.8 | 34 |
| 512 | 128256 | 2048 | y | 64 | 387.75 | 693.7 | 35 |
| 2048 | 2048 | 8192 |  | 256 | 78.38 | 876.7 | 44 |
| 2048 | 3072 | 2048 |  | 256 | 40.05 | 643.4 | 33 |
| 2048 | 2048 | 2048 |  | 256 | 27.73 | 619.5 | 31 |
| 1720 | 16384 | 2048 |  | 16 | 170.38 | 677.5 | 34 |
| 1720 | 2048 | 8192 |  | 16 | 77.31 | 746.5 | 38 |
| 1720 | 3072 | 2048 |  | 16 | 33.61 | 644.0 | 33 |
| 1720 | 2048 | 2048 |  | 16 | 27.34 | 527.8 | 27 |

**sc_4k_b4_eager: logical bytes per token** (one batch, 17164 real / 17244 padded tokens; eager ops' inputs + outputs, Triton bmm inputs + output)

| category | MB per real token | % |
|---|---|---|
| int GEMM, linear layers | 4.45 | 0.7 |
| int GEMM, lm_head | 1.96 | 0.3 |
| attention QK (bmm) | 9.95 | 1.5 |
| attention PV (bmm) | 9.95 | 1.5 |
| attention softmax (mask, max, shift, sum) | 225.88 | 35.0 |
| exp LUT | 124.45 | 19.3 |
| attention num/den divide (compiled: + quantization) | 2.65 | 0.4 |
| RMSNorm | 6.55 | 1.0 |
| RMSNorm + quantization (compiled segment) | 1.09 | 0.2 |
| quantization (quant_rows, limb split) | 77.70 | 12.0 |
| SiLU LUT | 44.77 | 6.9 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 13.70 | 2.1 |
| GEMM epilogue (limb join, rescale, residual add) | 109.82 | 17.0 |
| RoPE / q,k,v split | 6.29 | 1.0 |
| other elementwise | 0.54 | 0.1 |
| log-softmax float64 (scorer) | 5.88 | 0.9 |
| **total** | 645.64 | 100 |

### sc_4k_b8_eager: q2_4096_B.json, batch 8, eager, nl 2

unprofiled: 3781.1 tok/s over 2 batches (68947 tokens, sec [18.237, 18.232]); padding: {'real_tokens': 68947, 'computed_tokens': 69640, 'token_util': 0.99, 'real_qk_pairs': 148590962, 'computed_qk_pairs': 160461000, 'qk_util': 0.926}
profiled: 1 batches, 34424 tokens, wall 9.508 s, 254157 kernels (7.4 per token)

**sc_4k_b8_eager: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 180.82 | 1152 | 2.1 | 1.9 |
| int GEMM, lm_head | 49.96 | 128 | 0.6 | 0.5 |
| attention QK (bmm) | 1092.96 | 288 | 12.5 | 11.5 |
| attention PV (bmm) | 244.20 | 288 | 2.8 | 2.6 |
| attention softmax (mask, max, shift, sum) | 2395.40 | 4224 | 27.5 | 25.2 |
| exp LUT | 1269.66 | 1824 | 14.6 | 13.4 |
| attention num/den divide (compiled: + quantization) | 41.52 | 2016 | 0.5 | 0.4 |
| RMSNorm | 147.44 | 52800 | 1.7 | 1.6 |
| RMSNorm + quantization (compiled segment) | 22.44 | 6336 | 0.3 | 0.2 |
| quantization (quant_rows, limb split) | 1085.89 | 113920 | 12.5 | 11.4 |
| SiLU LUT | 489.27 | 5184 | 5.6 | 5.1 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 163.17 | 2880 | 1.9 | 1.7 |
| GEMM epilogue (limb join, rescale, residual add) | 1345.60 | 46656 | 15.4 | 14.2 |
| RoPE / q,k,v split | 77.64 | 10368 | 0.9 | 0.8 |
| other elementwise | 17.42 | 5453 | 0.2 | 0.2 |
| log-softmax float64 (scorer) | 96.43 | 640 | 1.1 | 1.0 |
| **sum of kernel time** | 8719.81 | 254157 | 100 | 91.7 |
| GPU busy (union of kernels) | 8719.81 | | | 91.7 |
| GPU idle: launch / Python / host sync | 788.19 | | | 8.3 |
| wall | 9508.00 | | | 100 |

**sc_4k_b8_eager: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 6750.13 | 246011 | 77.4 | 71.0 |
| attention (bmm q.k / p.v) | 1337.16 | 576 | 15.3 | 14.1 |
| eager reduction | 353.07 | 6162 | 4.0 | 3.7 |
| int GEMM (cuBLASLt _int_mm) | 230.78 | 1280 | 2.6 | 2.4 |
| other | 48.68 | 128 | 0.6 | 0.5 |
| GPU busy | 8719.81 | | | 91.7 |
| GPU idle | 788.19 | | | 8.3 |

**sc_4k_b8_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 4096 | 16384 | 2048 |  | 272 | 388.75 | 707.1 | 36 |
| 512 | 128256 | 2048 | y | 128 | 390.33 | 689.1 | 35 |
| 4096 | 2048 | 8192 |  | 272 | 144.9 | 948.5 | 48 |
| 4096 | 3072 | 2048 |  | 272 | 76.45 | 674.2 | 34 |
| 4096 | 2048 | 2048 |  | 272 | 51.65 | 665.3 | 34 |
| 64 | 2048 | 8192 |  | 16 | 19.41 | 110.6 | 6 |
| 64 | 16384 | 2048 |  | 16 | 16.94 | 253.6 | 13 |
| 64 | 3072 | 2048 |  | 16 | 7.85 | 102.6 | 5 |
| 64 | 2048 | 2048 |  | 16 | 7.03 | 76.4 | 4 |

### sc_gsm_b8_comp: q1_B.json, batch 8, compiled, nl 2

unprofiled: 27502.4 tok/s over 128 batches (430739 tokens, sec [21.052, 15.662, 15.661]); padding: {'real_tokens': 430739, 'computed_tokens': 476984, 'token_util': 0.903, 'real_qk_pairs': 94165159, 'computed_qk_pairs': 167534392, 'qk_util': 0.5621}
profiled: 4 batches, 13294 tokens, wall 0.51 s, 8253 kernels (0.6 per token)

**sc_gsm_b8_comp: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 78.28 | 512 | 17.1 | 15.3 |
| int GEMM, lm_head | 11.13 | 32 | 2.4 | 2.2 |
| attention QK (bmm) | 84.12 | 128 | 18.3 | 16.5 |
| attention PV (bmm) | 24.15 | 128 | 5.3 | 4.7 |
| attention softmax (mask, max, shift, sum) | 25.03 | 128 | 5.5 | 4.9 |
| attention num/den divide (compiled: + quantization) | 18.74 | 512 | 4.1 | 3.7 |
| RMSNorm + quantization (compiled segment) | 30.17 | 585 | 6.6 | 5.9 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 72.17 | 768 | 15.7 | 14.2 |
| GEMM epilogue (limb join, rescale, residual add) | 75.91 | 1632 | 16.5 | 14.9 |
| RoPE / q,k,v split | 10.86 | 1152 | 2.4 | 2.1 |
| other elementwise | 7.37 | 2516 | 1.6 | 1.4 |
| log-softmax float64 (scorer) | 20.82 | 160 | 4.5 | 4.1 |
| **sum of kernel time** | 458.76 | 8253 | 100 | 90.0 |
| GPU busy (union of kernels) | 458.76 | | | 90.0 |
| GPU idle: launch / Python / host sync | 51.24 | | | 10.0 |
| wall | 510.00 | | | 100 |

**sc_gsm_b8_comp: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| fused elementwise (Inductor) | 232.88 | 4777 | 50.8 | 45.7 |
| attention (bmm q.k / p.v) | 108.27 | 256 | 23.6 | 21.2 |
| int GEMM (cuBLASLt _int_mm) | 89.42 | 544 | 19.5 | 17.5 |
| eager elementwise | 17.19 | 2604 | 3.7 | 3.4 |
| other | 10.82 | 32 | 2.4 | 2.1 |
| eager reduction | 0.19 | 40 | 0.0 | 0.0 |
| GPU busy | 458.76 | | | 90.0 |
| GPU idle | 51.24 | | | 10.0 |

**sc_gsm_b8_comp: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 4096 | 16384 | 2048 |  | 64 | 393.54 | 698.5 | 35 |
| 4096 | 2048 | 8192 |  | 64 | 146.41 | 938.7 | 47 |
| 512 | 128256 | 2048 | y | 22 | 392.39 | 685.5 | 35 |
| 3440 | 16384 | 2048 |  | 16 | 332.89 | 693.5 | 35 |
| 3216 | 16384 | 2048 |  | 16 | 315.03 | 685.1 | 35 |
| 4096 | 3072 | 2048 |  | 64 | 77.41 | 665.8 | 34 |
| 2976 | 16384 | 2048 |  | 16 | 289.95 | 688.8 | 35 |
| 2848 | 16384 | 2048 |  | 16 | 276.77 | 690.6 | 35 |
| 4096 | 2048 | 2048 |  | 64 | 52.48 | 654.7 | 33 |
| 3440 | 2048 | 8192 |  | 16 | 145.4 | 793.9 | 40 |

### sc_gsm_b8_eager: q1_B.json, batch 8, eager, nl 2

unprofiled: 6674.7 tok/s over 128 batches (430739 tokens, sec [64.594, 64.516, 64.534]); padding: {'real_tokens': 430739, 'computed_tokens': 476984, 'token_util': 0.903, 'real_qk_pairs': 94165159, 'computed_qk_pairs': 167534392, 'qk_util': 0.5621}
profiled: 4 batches, 13294 tokens, wall 2.041 s, 107590 kernels (8.1 per token)

**sc_gsm_b8_eager: time by category (ms over the profiled batches)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 77.51 | 512 | 4.4 | 3.8 |
| int GEMM, lm_head | 11.07 | 32 | 0.6 | 0.5 |
| attention QK (bmm) | 78.98 | 128 | 4.4 | 3.9 |
| attention PV (bmm) | 23.54 | 128 | 1.3 | 1.2 |
| attention softmax (mask, max, shift, sum) | 150.29 | 1792 | 8.4 | 7.4 |
| exp LUT | 78.58 | 768 | 4.4 | 3.8 |
| attention num/den divide (compiled: + quantization) | 16.93 | 896 | 1.0 | 0.8 |
| RMSNorm | 60.64 | 21600 | 3.4 | 3.0 |
| RMSNorm + quantization (compiled segment) | 9.51 | 2592 | 0.5 | 0.5 |
| quantization (quant_rows, limb split) | 446.55 | 48434 | 25.1 | 21.9 |
| SiLU LUT | 203.29 | 2304 | 11.4 | 10.0 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 67.94 | 1280 | 3.8 | 3.3 |
| GEMM epilogue (limb join, rescale, residual add) | 495.56 | 19840 | 27.8 | 24.3 |
| RoPE / q,k,v split | 32.03 | 4608 | 1.8 | 1.6 |
| other elementwise | 7.43 | 2516 | 0.4 | 0.4 |
| log-softmax float64 (scorer) | 21.17 | 160 | 1.2 | 1.0 |
| **sum of kernel time** | 1781.02 | 107590 | 100 | 87.3 |
| GPU busy (union of kernels) | 1781.02 | | | 87.3 |
| GPU idle: launch / Python / host sync | 259.98 | | | 12.7 |
| wall | 2041.00 | | | 100 |

**sc_gsm_b8_eager: time by kernel class**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 1507.67 | 104158 | 84.7 | 73.9 |
| attention (bmm q.k / p.v) | 102.52 | 256 | 5.8 | 5.0 |
| int GEMM (cuBLASLt _int_mm) | 88.58 | 544 | 5.0 | 4.3 |
| eager reduction | 71.38 | 2600 | 4.0 | 3.5 |
| other | 10.86 | 32 | 0.6 | 0.5 |
| GPU busy | 1781.02 | | | 87.3 |
| GPU idle | 259.98 | | | 12.7 |

**sc_gsm_b8_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 4096 | 16384 | 2048 |  | 64 | 389.37 | 706.0 | 36 |
| 4096 | 2048 | 8192 |  | 64 | 145.25 | 946.2 | 48 |
| 512 | 128256 | 2048 | y | 22 | 389.77 | 690.1 | 35 |
| 3440 | 16384 | 2048 |  | 16 | 329.25 | 701.1 | 35 |
| 3216 | 16384 | 2048 |  | 16 | 310.67 | 694.7 | 35 |
| 4096 | 3072 | 2048 |  | 64 | 76.62 | 672.7 | 34 |
| 2976 | 16384 | 2048 |  | 16 | 287.95 | 693.6 | 35 |
| 2848 | 16384 | 2048 |  | 16 | 277.97 | 687.6 | 35 |
| 4096 | 2048 | 2048 |  | 64 | 51.69 | 664.8 | 34 |
| 3440 | 2048 | 8192 |  | 16 | 143.57 | 804.0 | 41 |

**sc_gsm_b8_eager: logical bytes per token** (one batch, 3300 real / 3768 padded tokens; eager ops' inputs + outputs, Triton bmm inputs + output)

| category | MB per real token | % |
|---|---|---|
| int GEMM, linear layers | 4.56 | 1.3 |
| int GEMM, lm_head | 1.13 | 0.3 |
| attention QK (bmm) | 2.01 | 0.6 |
| attention PV (bmm) | 2.01 | 0.6 |
| attention softmax (mask, max, shift, sum) | 39.84 | 11.2 |
| exp LUT | 21.95 | 6.2 |
| attention num/den divide (compiled: + quantization) | 3.01 | 0.8 |
| RMSNorm | 7.32 | 2.1 |
| RMSNorm + quantization (compiled segment) | 1.22 | 0.3 |
| quantization (quant_rows, limb split) | 87.90 | 24.7 |
| SiLU LUT | 50.88 | 14.3 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 15.56 | 4.4 |
| GEMM epilogue (limb join, rescale, residual add) | 108.02 | 30.3 |
| RoPE / q,k,v split | 7.15 | 2.0 |
| other elementwise | 0.61 | 0.2 |
| log-softmax float64 (scorer) | 2.95 | 0.8 |
| **total** | 356.13 | 100 |

## Decode step (outside vLLM)

### dec_B1_c2304_compiled: B 1, ctx 2304, compiled

{'ms_forward_nograph': 10.267, 'ms_step_nograph': 11.143, 'ms_tail': 0.642, 'ms_forward_graph': 2.536, 'ms_step_graph': 3.03}; profiled step: wall 4.352 ms, busy 2.516 ms, 569 kernels

**dec_B1_c2304_compiled: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| fused elementwise (Inductor) | 0.91 | 224 | 36.0 | 20.8 |
| int GEMM (cuBLASLt _int_mm) | 0.76 | 64 | 30.1 | 17.4 |
| eager elementwise | 0.36 | 234 | 14.3 | 8.3 |
| attention (fused paged) | 0.34 | 32 | 13.6 | 7.9 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.10 | 1 | 3.8 | 2.2 |
| fused elementwise (Inductor) | sampler | 0.03 | 5 | 1.0 | 0.6 |
| fused elementwise (Inductor) | lm_head | 0.02 | 4 | 0.6 | 0.4 |
| eager elementwise | sampler | 0.01 | 2 | 0.2 | 0.1 |
| other | sampler | 0.00 | 1 | 0.2 | 0.1 |
| eager reduction | 0.00 | 1 | 0.1 | 0.1 |
| eager elementwise | lm_head | 0.00 | 1 | 0.1 | 0.0 |
| GPU busy | 2.52 | | | 57.8 |
| GPU idle | 1.84 | | | 42.2 |

**dec_B1_c2304_compiled: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 24 | 128256 | 2048 | y | 5 | 96.2 | 131.1 | 7 |

### dec_B1_c2304_eager: B 1, ctx 2304, eager

{'ms_forward_nograph': 103.353, 'ms_step_nograph': 106.185, 'ms_tail': 3.156}; profiled step: wall 215.806 ms, busy 17.968 ms, 12686 kernels

**dec_B1_c2304_eager: time by category per step (ms)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 0.78 | 64 | 4.3 | 0.4 |
| int GEMM, lm_head | 0.10 | 1 | 0.5 | 0.0 |
| attention fused paged (QK+softmax+PV) | 0.71 | 256 | 4.0 | 0.3 |
| RMSNorm | 3.16 | 2475 | 17.6 | 1.5 |
| RMSNorm + quantization (compiled segment) | 0.39 | 297 | 2.2 | 0.2 |
| quantization (quant_rows, limb split) | 7.50 | 5931 | 41.7 | 3.5 |
| SiLU LUT | 0.46 | 288 | 2.6 | 0.2 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 0.25 | 160 | 1.4 | 0.1 |
| GEMM epilogue (limb join, rescale, residual add) | 3.36 | 2372 | 18.7 | 1.6 |
| RoPE / q,k,v split | 0.77 | 560 | 4.3 | 0.4 |
| other elementwise | 0.21 | 140 | 1.2 | 0.1 |
| sampler (Gumbel, argmax) | 0.28 | 142 | 1.6 | 0.1 |
| **sum of kernel time** | 17.97 | 12686 | 100 | 8.3 |
| GPU busy (union of kernels) | 17.97 | | | 8.3 |
| GPU idle: launch / Python / host sync | 197.84 | | | 91.7 |
| wall | 215.81 | | | 100 |

**dec_B1_c2304_eager: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 15.44 | 11962 | 85.9 | 7.2 |
| int GEMM (cuBLASLt _int_mm) | 0.78 | 64 | 4.3 | 0.4 |
| eager reduction | 0.72 | 273 | 4.0 | 0.3 |
| attention (fused paged) | 0.35 | 32 | 2.0 | 0.2 |
| eager elementwise | lm_head | 0.27 | 208 | 1.5 | 0.1 |
| eager elementwise | sampler | 0.25 | 139 | 1.4 | 0.1 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.10 | 1 | 0.5 | 0.0 |
| eager reduction | sampler | 0.03 | 2 | 0.2 | 0.0 |
| eager reduction | lm_head | 0.02 | 4 | 0.1 | 0.0 |
| other | sampler | 0.00 | 1 | 0.0 | 0.0 |
| GPU busy | 17.97 | | | 8.3 |
| GPU idle | 197.84 | | | 91.7 |

**dec_B1_c2304_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 24 | 2048 | 8192 |  | 80 | 18.88 | 42.6 | 2 |
| 24 | 16384 | 2048 |  | 80 | 15.69 | 102.7 | 5 |
| 24 | 2048 | 2048 |  | 80 | 7.01 | 28.7 | 1 |
| 24 | 3072 | 2048 |  | 80 | 6.98 | 43.2 | 2 |
| 24 | 128256 | 2048 | y | 5 | 96.52 | 130.6 | 7 |

**dec_B1_c2304_eager: logical bytes per token** (paged attention: K read twice + V once, int32)

| category | MB per token | % |
|---|---|---|
| int GEMM, linear layers | 1014.76 | 47.9 |
| int GEMM, lm_head | 275.03 | 13.0 |
| attention fused paged (QK+softmax+PV) | 229.47 | 10.8 |
| RMSNorm | 6.49 | 0.3 |
| RMSNorm + quantization (compiled segment) | 1.62 | 0.1 |
| quantization (quant_rows, limb split) | 93.96 | 4.4 |
| SiLU LUT | 44.56 | 2.1 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 13.63 | 0.6 |
| GEMM epilogue (limb join, rescale, residual add) | 114.71 | 5.4 |
| RoPE / q,k,v split | 5.73 | 0.3 |
| other elementwise | 2.67 | 0.1 |
| sampler (Gumbel, argmax) | 313.71 | 14.8 |
| **total** | 2116.35 | 100 |

### dec_B1_c512_compiled: B 1, ctx 512, compiled

{'ms_forward_nograph': 10.258, 'ms_step_nograph': 11.04, 'ms_tail': 0.628, 'ms_forward_graph': 2.4, 'ms_step_graph': 2.887}; profiled step: wall 4.293 ms, busy 2.38 ms, 569 kernels

**dec_B1_c512_compiled: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| fused elementwise (Inductor) | 0.90 | 224 | 38.0 | 21.0 |
| int GEMM (cuBLASLt _int_mm) | 0.76 | 64 | 32.0 | 17.7 |
| eager elementwise | 0.36 | 234 | 15.1 | 8.4 |
| attention (fused paged) | 0.21 | 32 | 8.7 | 4.8 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.10 | 1 | 4.0 | 2.2 |
| fused elementwise (Inductor) | sampler | 0.03 | 5 | 1.1 | 0.6 |
| fused elementwise (Inductor) | lm_head | 0.02 | 4 | 0.6 | 0.4 |
| eager elementwise | sampler | 0.01 | 2 | 0.2 | 0.1 |
| other | sampler | 0.00 | 1 | 0.2 | 0.1 |
| eager reduction | 0.00 | 1 | 0.1 | 0.1 |
| eager elementwise | lm_head | 0.00 | 1 | 0.1 | 0.0 |
| GPU busy | 2.38 | | | 55.4 |
| GPU idle | 1.91 | | | 44.6 |

**dec_B1_c512_compiled: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 24 | 128256 | 2048 | y | 5 | 95.88 | 131.5 | 7 |

### dec_B1_c512_eager: B 1, ctx 512, eager

{'ms_forward_nograph': 103.932, 'ms_step_nograph': 107.172, 'ms_tail': 3.197}; profiled step: wall 212.773 ms, busy 17.835 ms, 12686 kernels

**dec_B1_c512_eager: time by category per step (ms)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 0.78 | 64 | 4.4 | 0.4 |
| int GEMM, lm_head | 0.10 | 1 | 0.5 | 0.0 |
| attention fused paged (QK+softmax+PV) | 0.57 | 256 | 3.2 | 0.3 |
| RMSNorm | 3.16 | 2475 | 17.7 | 1.5 |
| RMSNorm + quantization (compiled segment) | 0.39 | 297 | 2.2 | 0.2 |
| quantization (quant_rows, limb split) | 7.50 | 5931 | 42.1 | 3.5 |
| SiLU LUT | 0.46 | 288 | 2.6 | 0.2 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 0.26 | 160 | 1.4 | 0.1 |
| GEMM epilogue (limb join, rescale, residual add) | 3.36 | 2372 | 18.8 | 1.6 |
| RoPE / q,k,v split | 0.77 | 560 | 4.3 | 0.4 |
| other elementwise | 0.21 | 140 | 1.2 | 0.1 |
| sampler (Gumbel, argmax) | 0.28 | 142 | 1.6 | 0.1 |
| **sum of kernel time** | 17.83 | 12686 | 100 | 8.4 |
| GPU busy (union of kernels) | 17.84 | | | 8.4 |
| GPU idle: launch / Python / host sync | 194.94 | | | 91.6 |
| wall | 212.77 | | | 100 |

**dec_B1_c512_eager: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 15.44 | 11962 | 86.6 | 7.3 |
| int GEMM (cuBLASLt _int_mm) | 0.78 | 64 | 4.4 | 0.4 |
| eager reduction | 0.73 | 273 | 4.1 | 0.3 |
| eager elementwise | lm_head | 0.27 | 208 | 1.5 | 0.1 |
| eager elementwise | sampler | 0.25 | 139 | 1.4 | 0.1 |
| attention (fused paged) | 0.21 | 32 | 1.2 | 0.1 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.10 | 1 | 0.5 | 0.0 |
| eager reduction | sampler | 0.03 | 2 | 0.2 | 0.0 |
| eager reduction | lm_head | 0.02 | 4 | 0.1 | 0.0 |
| other | sampler | 0.00 | 1 | 0.0 | 0.0 |
| GPU busy | 17.84 | | | 8.4 |
| GPU idle | 194.94 | | | 91.6 |

**dec_B1_c512_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 24 | 2048 | 8192 |  | 80 | 18.88 | 42.6 | 2 |
| 24 | 16384 | 2048 |  | 80 | 15.87 | 101.5 | 5 |
| 24 | 3072 | 2048 |  | 80 | 7.01 | 43.1 | 2 |
| 24 | 2048 | 2048 |  | 80 | 6.93 | 29.0 | 1 |
| 24 | 128256 | 2048 | y | 5 | 96.66 | 130.4 | 7 |

**dec_B1_c512_eager: logical bytes per token** (paged attention: K read twice + V once, int32)

| category | MB per token | % |
|---|---|---|
| int GEMM, linear layers | 1014.76 | 52.3 |
| int GEMM, lm_head | 275.03 | 14.2 |
| attention fused paged (QK+softmax+PV) | 53.31 | 2.7 |
| RMSNorm | 6.49 | 0.3 |
| RMSNorm + quantization (compiled segment) | 1.62 | 0.1 |
| quantization (quant_rows, limb split) | 93.96 | 4.8 |
| SiLU LUT | 44.56 | 2.3 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 13.63 | 0.7 |
| GEMM epilogue (limb join, rescale, residual add) | 114.71 | 5.9 |
| RoPE / q,k,v split | 5.73 | 0.3 |
| other elementwise | 2.67 | 0.1 |
| sampler (Gumbel, argmax) | 313.71 | 16.2 |
| **total** | 1940.19 | 100 |

### dec_B64_c2304_compiled: B 64, ctx 2304, compiled

{'ms_forward_nograph': 12.119, 'ms_step_nograph': 14.124, 'ms_tail': 1.848, 'ms_forward_graph': 10.89, 'ms_step_graph': 12.641}; profiled step: wall 14.068 ms, busy 11.37 ms, 810 kernels

**dec_B64_c2304_compiled: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| attention (fused paged) | 7.59 | 32 | 66.8 | 53.9 |
| fused elementwise (Inductor) | 1.79 | 464 | 15.7 | 12.7 |
| int GEMM (cuBLASLt _int_mm) | 0.85 | 64 | 7.5 | 6.0 |
| eager elementwise | 0.48 | 234 | 4.2 | 3.4 |
| fused elementwise (Inductor) | sampler | 0.36 | 5 | 3.2 | 2.6 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.13 | 1 | 1.1 | 0.9 |
| fused elementwise (Inductor) | lm_head | 0.09 | 5 | 0.8 | 0.6 |
| eager elementwise | sampler | 0.05 | 2 | 0.4 | 0.4 |
| other | sampler | 0.03 | 1 | 0.3 | 0.2 |
| eager reduction | 0.01 | 1 | 0.0 | 0.0 |
| eager elementwise | lm_head | 0.00 | 1 | 0.0 | 0.0 |
| GPU busy | 11.37 | | | 80.8 |
| GPU idle | 2.70 | | | 19.2 |

**dec_B64_c2304_compiled: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 128 | 128256 | 2048 | y | 5 | 128.65 | 522.7 | 26 |

### dec_B64_c2304_eager: B 64, ctx 2304, eager

{'ms_forward_nograph': 104.534, 'ms_step_nograph': 115.203, 'ms_tail': 10.304}; profiled step: wall 217.007 ms, busy 35.665 ms, 12556 kernels

**dec_B64_c2304_eager: time by category per step (ms)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 0.86 | 64 | 2.4 | 0.4 |
| int GEMM, lm_head | 0.13 | 1 | 0.4 | 0.1 |
| attention fused paged (QK+softmax+PV) | 7.93 | 256 | 22.2 | 3.7 |
| RMSNorm | 3.42 | 2475 | 9.6 | 1.6 |
| RMSNorm + quantization (compiled segment) | 0.41 | 297 | 1.2 | 0.2 |
| quantization (quant_rows, limb split) | 8.30 | 5801 | 23.3 | 3.8 |
| SiLU LUT | 0.86 | 288 | 2.4 | 0.4 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 0.49 | 160 | 1.4 | 0.2 |
| GEMM epilogue (limb join, rescale, residual add) | 5.13 | 2372 | 14.4 | 2.4 |
| RoPE / q,k,v split | 1.01 | 560 | 2.8 | 0.5 |
| other elementwise | 0.26 | 140 | 0.7 | 0.1 |
| sampler (Gumbel, argmax) | 6.86 | 142 | 19.2 | 3.2 |
| **sum of kernel time** | 35.67 | 12556 | 100 | 16.4 |
| GPU busy (union of kernels) | 35.66 | | | 16.4 |
| GPU idle: launch / Python / host sync | 181.34 | | | 83.6 |
| wall | 217.01 | | | 100 |

**dec_B64_c2304_eager: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 17.84 | 11834 | 50.0 | 8.2 |
| attention (fused paged) | 7.49 | 32 | 21.0 | 3.5 |
| eager elementwise | sampler | 6.73 | 139 | 18.9 | 3.1 |
| eager reduction | 1.59 | 273 | 4.5 | 0.7 |
| int GEMM (cuBLASLt _int_mm) | 0.86 | 64 | 2.4 | 0.4 |
| eager elementwise | lm_head | 0.84 | 206 | 2.4 | 0.4 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.13 | 1 | 0.4 | 0.1 |
| eager reduction | sampler | 0.09 | 2 | 0.2 | 0.0 |
| eager reduction | lm_head | 0.05 | 4 | 0.1 | 0.0 |
| other | sampler | 0.04 | 1 | 0.1 | 0.0 |
| GPU busy | 35.66 | | | 16.4 |
| GPU idle | 181.34 | | | 83.6 |

**dec_B64_c2304_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 128 | 2048 | 8192 |  | 80 | 20.16 | 213.1 | 11 |
| 128 | 16384 | 2048 |  | 80 | 19.06 | 450.6 | 23 |
| 128 | 128256 | 2048 | y | 5 | 129.79 | 518.1 | 26 |
| 128 | 3072 | 2048 |  | 80 | 7.67 | 210.1 | 11 |
| 128 | 2048 | 2048 |  | 80 | 7.12 | 150.8 | 8 |

**dec_B64_c2304_eager: logical bytes per token** (paged attention: K read twice + V once, int32)

| category | MB per token | % |
|---|---|---|
| int GEMM, linear layers | 18.68 | 2.3 |
| int GEMM, lm_head | 5.13 | 0.6 |
| attention fused paged (QK+softmax+PV) | 229.47 | 27.8 |
| RMSNorm | 6.53 | 0.8 |
| RMSNorm + quantization (compiled segment) | 1.09 | 0.1 |
| quantization (quant_rows, limb split) | 77.38 | 9.4 |
| SiLU LUT | 44.56 | 5.4 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 13.63 | 1.6 |
| GEMM epilogue (limb join, rescale, residual add) | 110.75 | 13.4 |
| RoPE / q,k,v split | 5.73 | 0.7 |
| other elementwise | 2.67 | 0.3 |
| sampler (Gumbel, argmax) | 310.68 | 37.6 |
| **total** | 826.32 | 100 |

### dec_B64_c512_compiled: B 64, ctx 512, compiled

{'ms_forward_nograph': 11.671, 'ms_step_nograph': 13.735, 'ms_tail': 1.847, 'ms_forward_graph': 5.211, 'ms_step_graph': 6.937}; profiled step: wall 8.299 ms, busy 5.683 ms, 810 kernels

**dec_B64_c512_compiled: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| attention (fused paged) | 1.90 | 32 | 33.4 | 22.9 |
| fused elementwise (Inductor) | 1.78 | 464 | 31.4 | 21.5 |
| int GEMM (cuBLASLt _int_mm) | 0.85 | 64 | 14.9 | 10.2 |
| eager elementwise | 0.48 | 234 | 8.4 | 5.8 |
| fused elementwise (Inductor) | sampler | 0.37 | 5 | 6.5 | 4.5 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.13 | 1 | 2.3 | 1.5 |
| fused elementwise (Inductor) | lm_head | 0.09 | 5 | 1.5 | 1.0 |
| eager elementwise | sampler | 0.05 | 2 | 0.9 | 0.6 |
| other | sampler | 0.03 | 1 | 0.6 | 0.4 |
| eager reduction | 0.01 | 1 | 0.1 | 0.1 |
| eager elementwise | lm_head | 0.00 | 1 | 0.0 | 0.0 |
| GPU busy | 5.68 | | | 68.5 |
| GPU idle | 2.62 | | | 31.5 |

**dec_B64_c512_compiled: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 128 | 128256 | 2048 | y | 5 | 128.28 | 524.2 | 26 |

### dec_B64_c512_eager: B 64, ctx 512, eager

{'ms_forward_nograph': 106.166, 'ms_step_nograph': 116.456, 'ms_tail': 10.396}; profiled step: wall 216.27 ms, busy 30.136 ms, 12556 kernels

**dec_B64_c512_eager: time by category per step (ms)**

| category | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| int GEMM, linear layers | 0.86 | 64 | 2.9 | 0.4 |
| int GEMM, lm_head | 0.13 | 1 | 0.4 | 0.1 |
| attention fused paged (QK+softmax+PV) | 2.34 | 256 | 7.8 | 1.1 |
| RMSNorm | 3.44 | 2475 | 11.4 | 1.6 |
| RMSNorm + quantization (compiled segment) | 0.41 | 297 | 1.4 | 0.2 |
| quantization (quant_rows, limb split) | 8.32 | 5801 | 27.6 | 3.8 |
| SiLU LUT | 0.86 | 288 | 2.9 | 0.4 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 0.49 | 160 | 1.6 | 0.2 |
| GEMM epilogue (limb join, rescale, residual add) | 5.13 | 2372 | 17.0 | 2.4 |
| RoPE / q,k,v split | 1.01 | 560 | 3.4 | 0.5 |
| other elementwise | 0.26 | 140 | 0.9 | 0.1 |
| sampler (Gumbel, argmax) | 6.87 | 142 | 22.8 | 3.2 |
| **sum of kernel time** | 30.14 | 12556 | 100 | 13.9 |
| GPU busy (union of kernels) | 30.14 | | | 13.9 |
| GPU idle: launch / Python / host sync | 186.13 | | | 86.1 |
| wall | 216.27 | | | 100 |

**dec_B64_c512_eager: time by kernel class per step**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| eager elementwise | 17.87 | 11834 | 59.3 | 8.3 |
| eager elementwise | sampler | 6.75 | 139 | 22.4 | 3.1 |
| attention (fused paged) | 1.91 | 32 | 6.3 | 0.9 |
| eager reduction | 1.59 | 273 | 5.3 | 0.7 |
| int GEMM (cuBLASLt _int_mm) | 0.86 | 64 | 2.9 | 0.4 |
| eager elementwise | lm_head | 0.84 | 206 | 2.8 | 0.4 |
| int GEMM (cuBLASLt _int_mm) | lm_head | 0.13 | 1 | 0.4 | 0.1 |
| eager reduction | sampler | 0.09 | 2 | 0.3 | 0.0 |
| eager reduction | lm_head | 0.05 | 4 | 0.2 | 0.0 |
| other | sampler | 0.04 | 1 | 0.1 | 0.0 |
| GPU busy | 30.14 | | | 13.9 |
| GPU idle | 186.13 | | | 86.1 |

**dec_B64_c512_eager: int8 GEMMs** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)

| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |
|---|---|---|---|---|---|---|---|
| 128 | 2048 | 8192 |  | 80 | 20.1 | 213.7 | 11 |
| 128 | 16384 | 2048 |  | 80 | 18.99 | 452.4 | 23 |
| 128 | 128256 | 2048 | y | 5 | 129.5 | 519.3 | 26 |
| 128 | 3072 | 2048 |  | 80 | 7.72 | 208.7 | 11 |
| 128 | 2048 | 2048 |  | 80 | 7.15 | 150.2 | 8 |

**dec_B64_c512_eager: logical bytes per token** (paged attention: K read twice + V once, int32)

| category | MB per token | % |
|---|---|---|
| int GEMM, linear layers | 18.68 | 2.9 |
| int GEMM, lm_head | 5.13 | 0.8 |
| attention fused paged (QK+softmax+PV) | 53.31 | 8.2 |
| RMSNorm | 6.53 | 1.0 |
| RMSNorm + quantization (compiled segment) | 1.09 | 0.2 |
| quantization (quant_rows, limb split) | 77.38 | 11.9 |
| SiLU LUT | 44.56 | 6.9 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 13.63 | 2.1 |
| GEMM epilogue (limb join, rescale, residual add) | 110.75 | 17.0 |
| RoPE / q,k,v split | 5.73 | 0.9 |
| other elementwise | 2.67 | 0.4 |
| sampler (Gumbel, argmax) | 310.68 | 47.8 |
| **total** | 650.15 | 100 |

## vLLM decode (torch.profiler)

### vLLM plugin decode B 1, 512 new tokens: 234.0 tok/s, 4.274 ms per step (wall, median of 3)

**vLLM B 1: kernel classes over one profiled generate of 128 new tokens (76219 kernels, 595 per step; wall 783.4 ms)**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| fused elementwise (Inductor) | 124.31 | 30064 | 39.5 | 15.9 |
| int GEMM (cuBLASLt _int_mm) | 109.68 | 8448 | 34.9 | 14.0 |
| eager elementwise | 49.31 | 32111 | 15.7 | 6.3 |
| attention (fused paged) | 26.18 | 4096 | 8.3 | 3.3 |
| other | 3.34 | 956 | 1.1 | 0.4 |
| eager reduction | 0.97 | 288 | 0.3 | 0.1 |
| index/copy | 0.45 | 128 | 0.1 | 0.1 |
| memcpy/fill | 0.17 | 128 | 0.1 | 0.0 |
| GPU busy | 314.40 | | | 40.1 |
| GPU idle | 468.96 | | | 59.9 |

### vLLM plugin decode B 64, 512 new tokens: 3534.8 tok/s, 18.106 ms per step (wall, median of 3)

**vLLM B 64: kernel classes over one profiled generate of 128 new tokens (114930 kernels, 898 per step; wall 2710.6 ms)**

| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |
|---|---|---|---|---|
| fused elementwise (Inductor) | 1598.23 | 68416 | 81.4 | 59.0 |
| attention (fused paged) | 152.13 | 4128 | 7.7 | 5.6 |
| int GEMM (cuBLASLt _int_mm) | 125.77 | 8448 | 6.4 | 4.6 |
| eager elementwise | 73.24 | 32319 | 3.7 | 2.7 |
| other | 12.27 | 1075 | 0.6 | 0.5 |
| eager reduction | 1.47 | 288 | 0.1 | 0.1 |
| index/copy | 0.59 | 128 | 0.0 | 0.0 |
| memcpy/fill | 0.21 | 128 | 0.0 | 0.0 |
| GPU busy | 1963.90 | | | 72.5 |
| GPU idle | 746.75 | | | 27.5 |

## Combined scorer shares
| category (% of wall) | sc_gsm_b8_eager | sc_gsm_b8_comp | sc_4k_b4_eager | sc_4k_b4_comp | sc_4k_b1_eager | sc_4k_b1_comp |
|---|---|---|---|---|---|---|
| int GEMM, linear layers | 3.8 | 15.3 | 1.8 | 6.7 | 0.8 | 3.5 |
| int GEMM, lm_head | 0.5 | 2.2 | 0.5 | 1.8 | 0.2 | 0.7 |
| attention QK (bmm) | 3.9 | 16.5 | 10.3 | 39.0 | 3.7 | 17.8 |
| attention PV (bmm) | 1.2 | 4.7 | 2.5 | 9.0 | 1.0 | 4.3 |
| attention softmax (mask, max, shift, sum) | 7.4 | 4.9 | 23.0 | 11.7 | 7.9 | 4.7 |
| exp LUT | 3.8 | 0.0 | 12.2 | 0.0 | 4.2 | 0.0 |
| attention num/den divide (compiled: + quantization) | 0.8 | 3.7 | 0.4 | 1.7 | 0.2 | 0.8 |
| RMSNorm | 3.0 | 0.0 | 1.7 | 0.0 | 1.7 | 0.0 |
| RMSNorm + quantization (compiled segment) | 0.5 | 5.9 | 0.3 | 2.7 | 0.2 | 1.2 |
| quantization (quant_rows, limb split) | 21.9 | 0.0 | 10.8 | 0.0 | 5.3 | 0.0 |
| SiLU LUT | 10.0 | 0.0 | 4.9 | 0.0 | 1.4 | 0.0 |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 3.3 | 14.2 | 1.6 | 6.2 | 0.5 | 2.1 |
| GEMM epilogue (limb join, rescale, residual add) | 24.3 | 14.9 | 13.0 | 7.4 | 5.2 | 3.6 |
| RoPE / q,k,v split | 1.6 | 2.1 | 0.8 | 1.0 | 0.6 | 0.9 |
| other elementwise | 0.4 | 1.4 | 0.3 | 0.9 | 0.2 | 1.0 |
| log-softmax float64 (scorer) | 1.0 | 4.1 | 0.9 | 3.4 | 0.3 | 1.3 |
| GPU idle (launch, Python, host sync) | 12.7 | 10.0 | 15.1 | 8.6 | 66.5 | 58.2 |
| scorer tok/s (unprofiled) | 6,675 | 27,502 | 3,637 | 12,920 | 2,416 | 8,936 |
