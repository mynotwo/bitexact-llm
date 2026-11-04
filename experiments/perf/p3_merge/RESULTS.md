# p3_merge: perf3/merge (host + fuse + splitkv + 8b + spec) against main cc523a8

## Conclusion
1. One merge bug found and fixed (commit d306628). `graph._decode_fused` (fuse lane) used `paged.G`, which the 8b lane removed, so pytest failed (2 of 43: `test_decode_eager`, `test_decode_cuda_graph`) and the default `BITEXACT_FUSE=1` path would have crashed. Fix: use `m.H // m.HKV`, and take the fused path only for the 1B geometry (tri_fuse kernels are compiled with the 1B module constants H/HKV/HD/D/FF), so 8B runs the unfused path. No other code touched. Everything below ran on 7c79e4f (= d306628 + runner); a first launch on 0b270c6 was aborted at the pytest failure.
2. Bit identity: all PASS (list below). The `IntSampler.apply` conflict resolution is correct: spec's multi-row branch plus host's numpy path, covered by the spec k=0/k=4 checks and the suite.
3. 1B tok/s (median of 5, alternating base/merge): merge is faster or equal in every cell. T=0 B=1 +17%, B=32 +21%, B=128 +1.3%; T=1 B=1 +14%, B=32 +21%, B=128 +41%. Long context S=16k/32k: -3 to -19% ms/step.
4. 8B: plugin equals stock at B=1 (152 vs 152), is 11% behind stock at B=32, 41% behind at B=128 (8,250 vs 14,054), and 22% behind vLLM BATCH_INVARIANT+TRITON_ATTN at B=128 (10,635). It is 3x faster than BI+TRITON at B=1 and 2.3x at B=32.
5. Slower anywhere: no lane is slower than base on 1B in any median. Spec at B=32 is slower than k=0 (per the p3_spec lane; off by default). Profiler: wall-minus-busy grows 0.62 -> 1.19 ms at T=0 B=128 (host gaps are now exposed after the kernels shrank), which is why wall improves 5.6% while kernel time improves 23%.

## Bit identity (all on 141; base = cc523a8 archive, merge = 7c79e4f archive)
| check | result |
|---|---|
| GPU pytest (test_gpu_kernels, test_vllm_kernels, test_tri_fuse, test_int8_ops, test_decode_paths, test_overflow_check) | PASS, 41 passed, 1 skipped (after fix; 39 passed, 2 failed before) |
| 14-run suite, base and merge | PASS, all 14 runs token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0, compare_exit 0, compare_long_exit 0; 14 of 14 hash files `cmp` identical base vs merge (0 DIFF) |
| p2_sample token dumps (128 req; mixed T 0/1/0.7/0.5/1.5/2; T=0; logprobs=1) | PASS, 3 of 3 IDENTICAL |
| spec check, k=0 vs k=4 (tests/vllm/spec_check.py, vLLM graphs) | PASS, 9 of 9 token-identical: suite (12 mixed targets), rep and bench workloads at B=1 and 32, T=0 and 0.7. Tokens only (no logit hashes) |
| 8B batch invariance (p3_8b/batch_inv.py, alone vs batch 32 vs 128, 192 hashed rows) | PASS graphs, PASS eager |
| 8B plugin vs standalone `IntLlama` 8B reference (ref + 7 suite runs: alone_pc1, busy_pc1_s0, busy_pc0_s1, busy_pc1_s2_b64, busy_eager_s4, busy_vllmgraph_s5, alone_vllmgraph; GPU_UTIL=0.6) | PASS, compare_exit 0, 576 positions per run, token_seqs_diff 0, logit_hash_diff 0 |
| 1B longctx S=16k, 32k, B=1 and 8 | runs complete (ms/step below). The 5k/9k/14k long-prompt suite jobs are in the 14-run suite |

## Long context, decode ms/step (CUDA graph, longctx_step.py)
| S | B | base | merge |
|---|---|---|---|
| 16k | 1 | 2.379 | 1.971 |
| 16k | 8 | 6.582 | 5.211 |
| 32k | 1 | 2.959 | 2.454 |
| 32k | 8 | 10.896 | 8.855 |

## 1B bench, GPU 7, 128 in / 128 out, output tok/s, median [min-max] of 5 reps, rep-wise alternating base/merge/vLLM rows in one session
| T | B | base cc523a8 | merge | ratio | vLLM bf16 stock | vLLM BATCH_INVARIANT + TRITON_ATTN |
|---|---|---|---|---|---|---|
| 0 | 1 | 451 [446-453] | 529 [527-535] | 1.17 | 610 [606-616] | 215 [214-215] |
| 0 | 32 | 11,038 [11,014-11,204] | 13,330 [13,020-13,413] | 1.21 | 16,148 [15,764-16,231] | 7,522 [7,518-7,589] |
| 0 | 128 | 29,190 [28,824-29,231] | 29,569 [29,263-29,900] | 1.01 | 33,525 [33,189-34,282] | 34,825 [34,591-34,907] |
| 1 | 1 | 447 [444-448] | 510 [508-524] | 1.14 | | |
| 1 | 32 | 10,087 [9,982-10,300] | 12,224 [12,027-12,352] | 1.21 | | |
| 1 | 128 | 20,319 [19,917-20,391] | 28,688 [24,514-28,868] | 1.41 | | |
vLLM rows are T=0 only (bench_attn.py). Ranges are tight (spread under 3% except merge T=1 B=128, one slow rep at 24.5k). Merge at B=128 T=0 is 85% of BI+TRITON (34.8k), 88% of stock. Host: 192-core shared box, 1-min load at the 185 bench launches min 5.7, median 10.0, max 128.5; GPU 7 was 0 MiB with no compute process at every launch (`out/quiet_check.log` has uptime and all-GPU memory per launch).

## 8B bench, same GPU 7, B in/out 128/128, T=0, tok/s median [min-max], 3 reps
| B | plugin int8 8B | vLLM bf16 stock | vLLM BATCH_INVARIANT + TRITON_ATTN |
|---|---|---|---|
| 1 | 152 [126-153] | 152 [152-152] | 50 [50-50] |
| 32 | 4,025 [2,932-4,027] | 4,510 [4,508-4,537] | 1,751 [1,748-1,754] |
| 128 | 8,250 [8,222-8,250] | 14,054 [13,979-14,056] | 10,635 [10,630-10,675] |
One slow plugin rep at B=1 (126) and B=32 (2,932); the other two reps are within 1%. The BI row sets TRITON_ATTN (the earlier p3_8b BI number did not: 9,995 at B=128 there). 8B uses the unfused decode path (fuse is 1B-geometry only), so the 1B fuse gain does not apply.

## Spec, B=1, k=0 vs k=4 (ngram, prompt_lookup 2..5), 3 reps, tok/s median [min-max]
| workload | T | k=0 | k=4 | ratio |
|---|---|---|---|---|
| rep (copy-heavy) | 0 | 540 [537-544] | 907 [730-922] | 1.68 |
| rep | 0.7 | 502 [501-514] | 975 [957-981] | 1.94 |
| bench (random prompts) | 0 | 533 [521-537] | 853 [745-859] | 1.60 |
| bench | 0.7 | 490 [458-501] | 777 [718-779] | 1.59 |
B=1 k=0 here (533 at T=0) matches the fused path. B=32 spec throughput was not re-benchmarked; p3_spec measured 0.4-0.9x there.

## Profile, B=128 decode, ms/step (median of 3 runs per tree, prof_vllm.py, interleaved base/merge)
| bucket | T=0 base | T=0 merge | delta | T=1 base | T=1 merge | delta |
|---|---|---|---|---|---|---|
| wall | 4.118 | 3.886 | -0.232 | 6.605 | 4.278 | -2.327 |
| kernel_sum (= busy) | 3.501 | 2.706 | -0.795 | 3.863 | 3.072 | -0.791 |
| wall - busy | 0.621 | 1.187 | +0.566 | 2.742 | 1.210 | -1.532 |
| elementwise | 1.115 | 0.245 | -0.870 | 1.115 | 0.246 | -0.869 |
| attention | 0.697 | 0.685 | -0.012 | 0.695 | 0.685 | -0.010 |
| sampling_logits | 0.434 | 0.392 | -0.042 | 0.800 | 0.761 | -0.039 |
| other (GEMMs, fused kernels) | 1.253 | 1.379 | +0.126 | 1.251 | 1.377 | +0.126 |
Fusion removes 0.87 ms of elementwise kernels (new fused kernels land in other). Attention is unchanged at R=128 (splitkv does not touch it). T=1 wall drops 2.3 ms because host sampler work (host lane) no longer stalls the stream. The profiler's own tok/s column is not comparable across runs (profiling overhead).

## Not done / not verified
- 8B profiler, 8B at 5 reps (3 reps), spec at B=32 and B=128 in this session, spec logit-hash comparison.
- The remaining B=128 T=0 gap to vLLM is host wall-minus-busy (1.19 ms) and GEMM; not investigated here.
- Fused decode for 8B geometry (tri_fuse is hard-wired to 1B constants): follow-up.

Logs and outputs: /data41/audreymao/p3_merge/out (suite, tokens, spec, s8, bi8, longctx, bench, bench8, specbench, prof0, prof1, quiet_check.log). Runner: experiments/perf/p3_merge/run.sh (stages pytest suite suitecmp tokens tokcmp spec s8 bi8 longctx bench bench8 specbench prof). The first spec launch had a shell variable bug in the runner (loop variable `w` shadowed the GPU-wait counter); fixed before the reported spec run.
