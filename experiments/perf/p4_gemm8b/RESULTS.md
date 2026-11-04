# perf4 / gemm8b: tuned skinny int8 GEMM for Llama-3.1-8B decode

Conclusion: bit-identity PASS. 8B B=128 +8.4% tok/s (8,215 -> 8,908, median of 3), B=32 +2.0%, B=1 +4% (noise-level); kernel time at B=128 decode 13.39 -> 12.46 ms/step (-0.93 ms, -7%). Gain is limited because at B=128 the GEMM rows are 256 (2 limbs x 128) and the biggest matrix (gate_up 28672x4096) has no Triton win at 256 rows, so it stays on cutlass.

## What changed
`src/bitexact/int8/gemm_skinny.py` TABLE gets the 8B shapes (6144x4096, 4096x4096, 4096x14336, 28672x4096, LM head 128256x4096), row buckets 24/64/128/256. A bucket value `None` means "torch._int_mm is as fast or faster" and falls back. 1B entries are untouched (keys differ, so 1B configs are byte-identical). Configs found offline by `search8b.py` (CUDA graph, HBM-cold rotating weights, grid BM {16,32 | 32,64,128} x BN {32..256} x BK {128,256} x split-K {1,2,4,8} x warps {4,8} x stages {3,4,5}); every candidate checked `torch.equal` against `_int_mm` during the search. Raw: `search0.json`. `tests/test_gemm_skinny.py` now covers all TABLE shapes (set union with LM 2048) and adds R=100.

## Microbench (H800 GPU 0, CUDA graph, cold weights, us; rows are the stacked-limb row count)
| shape (N,K) | rows | _int_mm | Triton best | used | roofline |
|---|---|---|---|---|---|
| wqkv 6144x4096 | 24 / 64 / 128 / 256 | 12.9 / 12.9 / 20.1 / 27.1 | 11.7 / 14.3 / 14.7 / 20.0 | 11.7 / 12.9 / 14.7 / 20.0 | 7.5 |
| wo 4096x4096 | 24 / 64 / 128 / 256 | 11.8 / 11.9 / 12.0 / 18.5 | 9.1 / 10.5 / 10.3 / 14.7 | same as Triton | 5.0 |
| wd 4096x14336 | 24 / 64 / 128 / 256 | 32.1 / 33.2 / 33.6 / 66.2 | 24.5 / 28.6 / 27.9 / 43.7 | same as Triton | 17.5 |
| wgu 28672x4096 | 24 / 64 / 128 / 256 | 44.8 / 47.3 / 65.2 / 93.2 | 43.1 / 49.4 / 51.9 / 93.8 | 44.8 / 47.3 / 51.9 / 93.2 | 35.1 |
| LM 128256x4096 | 24 / 64 / 128 / 256 | 186.5 / 286.1 / 294.9 / 338.6 | 179.7 / 358.4 / 214.2 / 420.2 | 186.5 / 286.1 / 214.2 / 338.6 | 156.8 |

Triton wins by >5% only where "used" differs from _int_mm; ties (wgu 24: 43.1 vs 44.8; lm 24) stay on _int_mm. LM head: Triton only at 65..128 rows (B=64 decode); at 256 rows (B=128) it loses, so unchanged.

## Bit-identity: PASS
- pytest tests/test_gemm_skinny.py + the p3_merge pytest stage: 41 passed, 1 skipped (Triton == _int_mm on every TABLE shape, 11 row counts, random/-128/127 extremes).
- 8B: branch vs main (cutlass-only) via `run_ref.py` (standalone IntLlama 8B): `compare.py` base_ref vs branch ref, alone_pc1, busy_pc1_s0, alone_vllmgraph: token_seqs_diff 0, logit_hash_diff 0, 576 positions each, 12 targets. Branch plugin vs its own standalone ref in the 7 s8 runs (alone/busy/pc0/budget64/eager/vllm-graph): all 0 diffs.
- 1B 14-job suite (`suite` + `suitecmp`, base 6d42e87 vs branch): all `cmp OK`, token_seqs_diff 0, incl. long runs.

## tok/s, 8B, T=0, 128 in / 128 out, H800 GPU 0, 3 runs each, median (min..max)
| B | base 6d42e87 | branch | delta |
|---|---|---|---|
| 1 | 151.6 (150.9..151.7) | 158.1 (154.2..158.5) | +4.3% |
| 32 | 3,956 (3,910..4,023) | 4,036 (4,031..4,232) | +2.0% |
| 128 | 8,215 (8,210..8,325) | 8,908 (8,884..8,910) | +8.4% |

B=1/32 shared-host noise is a few %, so only B=128 is a clear gain. For reference (brief): vLLM stock 152 / 4,510 / 14,054.

## Profile, 8B B=128 decode, ms/step (median of 3 profiles each)
| | base | branch |
|---|---|---|
| kernel_sum | 13.39 | 12.45 |
| cutlass gemm_int8 | 6.47 | 2.90 (wgu 256 rows, 2.1 ms, + LM head 0.33 stay) |
| Triton `_k` (qkv/wo/wd) | 0 | 2.65 |
| tok/s (profiled run) | 8,214 | 8,692 |

Net GEMM 6.47 -> 5.55 ms (qkv/wo/wd moved, 3 x 32 launches). Remaining step: attention 2.67, gate_up cutlass 2.1, `_h_kernel` 0.78.

## Didn't work / not verified
- gate_up at 256 rows: best Triton 93.8 us vs cutlass 93.2 us (roofline 35.1). Within the searched grid there is no win; this is the largest remaining GEMM item (2.1 ms/step at B=128). A wider grid (BM 256, BK 64, 2..6 stages, 16 warps; `EXT=1 ROWS=256 search8b.py wgu`, added after the run) was not executed.
- LM head: Triton only helps at 65..128 rows, which the B=128 decode (256 rows) never hits.
- Not verified: table is tuned on H800 only; other GPUs are correct (exact integers) but untuned. Row counts between buckets use the next bucket's config (not individually timed). 8B long-context and T=1 not rerun. B=1/32 gains are inside noise.
- The 8B graph decode still takes the unfused path (separate lane).
