# p3_fuse: fused decode-step kernels (add+norm, qkv+RoPE+KV store, attn-out quant, MLP flags)

## Conclusion
1. **Bit-identity: PASS.** Branch perf3/fuse vs base 07aa43c, GPU 7: 14 suite runs token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0 against run_ref; all 14 hash files byte-identical (`cmp OK`; two base runs, busy_pc1_s2_b64 and busy_eager_s4, were rerun after vLLM memory-profiling failures caused by another job on the GPU). GPU pytest: 40 passed, 1 skipped (includes new tests/test_tri_fuse.py, 12 cases, all kernels vs eager segments incl. bound flags).
2. **Kernels per decode step 486 -> 206; per layer about 28 -> 11** (B=128, vLLM graph, profiler).
3. **B=128 profile (ms/step, same session, GPU 7):** kernel sum 3.432 -> 2.719 (-21%); elementwise bucket 1.113 -> 0.246 (new fused kernels sit in "other"; attention 0.697 -> 0.685; gemm unchanged). Wall 3.867 -> 3.807 ms/step only (-1.6%): GPU busy fell 0.71 ms but wall-busy (host/launch gaps outside the graph) rose 0.44 -> 1.09 ms, so most of the kernel saving is not yet visible in wall time. Long context S=16k: B=1 2.379 -> 2.073 ms, B=8 6.587 -> 5.915 ms.
4. **tok/s T=0 (bench.py, 3 reps interleaved, noisy host, medians):** B=1 479 -> 533, B=32 10,992 -> 12,820, B=128 29,053 -> 30,133 (+3.7%). Spread per cell is 10-20% (other lanes on the host; a rep per cell failed or hit contention), so only the profiler numbers are reliable; final tok/s needs a quiet run.

## Kernels
Per layer before: rmsnorm/quant 2-3 Inductor kernels, post+RoPE+flags about 8, _store_kernel, attn-out about 4, post_add+norm about 5, MLP _h/_zero/_quant/amax 4, plus 4 GEMMs and attention. After (11): `_add_norm_kernel` (residual add of the previous GEMM + RMSNorm + 2-limb int8 quant, one row per program), qkv GEMM, `_qkv_kernel` (post, RoPE, bound flags, K/V limb planes stored straight into the paged cache), `_dec_acc_kernel`, `_attn_out_kernel` (rdiv + quant), wo GEMM, `_add_norm_kernel`, gate_up GEMM, `_h_kernel`, `_quant_kernel` (flag maxima via atomic max, flags evaluated once per step in `_flag_kernel`), down GEMM. Embedding gather is fused into the first norm.

## Implementation
`vllm_plugin/tri_fuse.py` (new), `graph.decode_forward` takes the fused path when `BITEXACT_FUSE=1` (default), nl=2, CUDA and no run-time post_check; otherwise the old path. `tri_mlp.silu_quant` gained an optional `scr` argument. Integer rules unchanged: same int64 operations in the same order, rdiv32 float32 estimate plus two exact corrections, isqrt via float64 plus corrections, row maxima exact in any order.

## Didn't work / not done
- GEMM-prologue fusion (RMSNorm+quant inside gemm_skinny): not tried. The profile shows the remaining non-GEMM kernels at 0.25 ms elementwise plus about 0.45 ms h/quant/qkv/attn_out; the larger loss is now wall-busy (host gaps), not kernel time.
- Wall gain is small (1.6% in profile); below the 3% threshold on the profiler's wall metric, 3.7% on noisy bench. Merge decision is the caller's.

## Not verified
- Quiet-machine tok/s; wall-busy increase cause (not investigated: graph replay should hide launches, possibly sampler/host work now exposed).
- Outputs on 141: /data41/audreymao/perf3_20261103/fuse/out2 (suite, pytest, partial bench) and out3 (bench, longctx, prof, redo).
