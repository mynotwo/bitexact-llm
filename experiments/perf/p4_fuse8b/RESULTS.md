# p4_fuse8b: fused decode kernels for the 8B geometry

Branch perf4/fuse8b, base main 6d42e87, H800 on 141, GPU 3 (GPU 1 stayed busy with other lanes), runner `run.sh` (this directory).

## Conclusion
- Bit identity: PASS. 1B suite (12 targets x 10 runs + 3 long targets x 4 runs, 576 / 48 positions per run): every `.hash` file cmp-identical to base 6d42e87 (14 of 14), token JSONs (mixed, T=0, logprobs) identical. 8B: 8 of 8 suite runs (ref, alone, busy, prefix cache on/off, budget 64, eager, vLLM graphs) match the standalone IntLlama 8B reference (compare_exit 0, 0 token and 0 logit-hash diffs, 576 positions each); batch invariance alone vs 32 vs 128, graphs and eager: PASS, 192 rows each.
- `graph.decode_forward` no longer checks for the 1B geometry. The fused path is taken whenever D is a power of two, H*HD == D, TRI_MLP, nl == 2, CUDA, no post_check (8B qualifies).
- 8B decode at B=128: elementwise bucket 2.56 ms to 0.76 ms per step, step wall 14.0 ms to 12.6 ms, kernels per step about 931 to 382.
- 8B tok/s (T=0, 128 in / 128 out, mean of 3 runs): B=1 151.9 to 163.3 (+7.5%), B=32 3,982 to 4,606 (+15.7%), B=128 8,220 to 9,267 (+12.7%).

## Change
`src/bitexact/vllm_plugin/tri_fuse.py`: the module constants D, H, HKV, HD are gone. `_qkv_kernel` takes HC, HKVC, HDC, GC as constexpr args from `m.H/HKV/HD`; `_attn_out_kernel` takes K2 (= H*HD from the num tensor) and HDC; `_run` uses `m.D`. The head-dim-128 RoPE tables and Llama-3.1 scaling come from `m.cos/m.sin` (already built from the constants), so the kernels needed no other change; nothing in the qkv, requant or add_norm paths depended on HD 64 or D 2048 beyond those constants (xs*xs sums stay below 2^63 at D=4096: 2^48 * 4096). The score rule lives in attention and is untouched.
`src/bitexact/vllm_plugin/graph.py`: geometry check replaced as above.
`tests/test_tri_fuse.py`: the module fixture is parametrized 1B / 8B (8B needs `BITEXACT_INT8_CONST_8B`, skipped otherwise); shapes come from the model. Every fused kernel (add_norm, embed_norm, qkv + KV store, attn_out, mlp_mid + flag, final bound flag) is compared by exact equality with the unfused segments: 26 passed (13 per geometry). Full pytest stage: 54 passed, 1 skipped.

## 8B throughput, tok/s per run
| B | base | branch |
|---|---|---|
| 1 | 152.5 / 151.2 / 152.0 | 163.1 / 163.5 / 163.2 |
| 32 | 3988.6 / 3975.7 / 3982.2 | 4596.5 / 4613.1 / 4607.5 |
| 128 | 8231.7 / 8239.1 / 8190.5 | 9333.1 / 9216.2 / 9251.3 |

## 8B profile, B=128 decode (ms per step, 2 runs each)
| | base | branch |
|---|---|---|
| step wall | 14.09 / 13.96 | 12.68 / 12.52 |
| kernel sum | 13.36 / 13.32 | 11.89 / 11.95 |
| gemm_int8 | 6.45 / 6.44 | 6.45 / 6.48 |
| elementwise bucket | 2.561 / 2.561 | 0.755 / 0.762 |
Kernels per step (trace kernel events / 16 profiled steps): base about 931, branch about 382. Branch fused kernels per step: `_add_norm_kernel` x65 0.74 ms, `_attn_out_kernel` x32 0.28 ms, `_qkv_kernel` x32 0.19 ms. The GEMMs (cutlass `_int_mm`, no 8B skinny TABLE entries) and attention stay the same; they are the remaining cost.

## Not done / not verified
- Base 8B (unfused path) was not run through the s8 suite; the branch is compared against the IntLlama reference, not against base 8B hashes.
- 8B with speculative decoding, long-context 8B, 1B throughput (not re-benchmarked; the 1B path runs the same kernels with constexpr geometry): not run.
- Kernels-per-step counts are derived from the trace event count divided by 16 (the profiler default for decode steps), so they include a few non-compute events.
- GPU 1 stayed occupied by other jobs, so everything ran serially on GPU 3; 3 reps for tok/s, 2 for the profile.
