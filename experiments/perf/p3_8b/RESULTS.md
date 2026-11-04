# p3_8b: Llama-3.1-8B in the vLLM plugin (int8 mode)

Branch perf3/8b (base 07aa43c), H800 on 141, GPU 0 shared with another lane.

## Conclusion
- The plugin runs 8B. Bit identity: 7 of 7 suite runs (short targets, alone / busy / prefix cache on and off / preemption-free budget 64 / eager / vLLM graphs) match the standalone int8 `IntLlama` 8B reference in tokens and in every int64 logits hash (576 positions each, 0 diffs). Alone vs batch 32 vs batch 128 (graphs and eager): PASS (192 hashed rows each).
- Same code on 1B: ref hashes of base and branch identical, 19 GPU/decode-path tests pass (base 18).
- Throughput: 8B plugin is slower than stock bf16 at B=32 and B=128, slower than BATCH_INVARIANT at B=128 (8,064 vs 9,995; that BATCH_INVARIANT run used the default attention backend, not TRITON_ATTN; measurement caveat below).

## What changed
- No 1B constants left in int8/plugin path: H, HKV, HD, D, FF come from the constants config (`llama.dims`); `Segs`, `qkv`, `attn_out`, `silu_mul`, `_rope`, KV-cache layout, plugin Attention declaration (HKV heads, head_size 2*HD), paged planner and all tri_paged kernels take them from tensor shapes. RoPE tables are built from the HF config (theta 500000, llama3 scaling factor 8) in the constants and `_extend_rope`.
- head_dim 128 score scale: z = rsh(max(d, -2^38) * 11863283, 24 + 2FQ + 3 - FA), SC128 = round(2^24/sqrt 2) (11863283.203). The clamp keeps |d*SC128| < 2^62; values below it are under the exp table (z < -32) both ways. Eager `softmax` and the Triton kernels (`_zscale`) use the same rule; head_dim 64 unchanged (`score_rule`). Note this is the int8 engine's own rule, not the exact engine's constant 273841598.
- Constants: `build_constants.py --mode int8` on 8B, sha256 a54d3be9...ef695df (9.6 GB, 32 layers).

## Load-time bounds at 8B (post_bound, all layers, nl=2)
| matrix | shape | max log2 bound on t | max row L1 | int32 acc use K*127*128 |
|---|---|---|---|---|
| wqkv | 6144x4096 | 31.68 | 146320 | 3.1% |
| wo | 4096x4096 | 31.58 | 132454 | 3.1% |
| wgu | 28672x4096 | 32.35 | 157102 | 3.1% |
| wd | 4096x14336 | 33.08 | 429895 | 10.9% (K=14336 fine) |
| lm | 128256x4096 | 30.41 | | |
All below the 2^38 limit, so `post_check` is False (no run-time check) as for 1B; l1 < 2^24 and |m| < 2^36 asserts hold. No bound failed. Attention int32 partials: q.K per dot <= 128*2^14, p.V < 43690 keys asserted; unchanged.

## Triton kernels at 8B shapes
All run on H800 without OutOfResources: attention fused decode (DH=128, BN 64/128), prefill (_max/_acc), tri_mlp (FF=14336), skinny GEMM (shapes not in TABLE fall back to torch._int_mm: all 8B shapes, so 8B uses cutlass for every linear, no tuned skinny configs), tri_sample (V=128256, same). Not verified: sm_89 shared memory for the head_dim=128 plugin attention (xhw_8b notes the exact engine's attention needed a fix there).
Plugin test `test_fused_paged_attention_exact` now parametrized HD in {64, 128} against the unfused engine: pass.

## Bench (B in / out = 128 / 128, T=0, output tok/s, 3 runs)
| B | plugin int8 8B | vLLM bf16 | vLLM BATCH_INVARIANT |
|---|---|---|---|
| 1 | 155.8 (1 run) | 152.0 / 152.2 / 151.6 | 49.5 / 31.7 / 49.3 |
| 32 | 2831 / 3800 / 4010 | 4511 / 4495 / 4503 | 1670 / 1662 / 1679 |
| 128 | 8060 / 8130 / 8064 | 14141 / 13993 / 14101 | 9995 / 9971 / 10098 |
Caveat: GPU 0 was shared with another lane (vLLM start of int B1 r2/r3 failed on memory; B=32 int runs vary 2831 to 4010), and the BI B=1 run varies 32 to 50. Treat B=1 and B=32 int numbers as unreliable; B=128 is stable. Quiet combined run still needed. ms/step profile at 8B not taken.

## Not done / not verified
- Standalone int8 8B reference is the existing `IntLlama` (no other int8 8B path exists); there is no comparison against the exact-mode engine (different model, int8 weights).
- Long-context suite (5k/9k/14k), 14-run suite items long_*, preemption run (kv-blocks 160), 8B profiler: not run.
- Suite used GPU_UTIL=0.6 for two runs (default 0.35 leaves no KV room next to 16 GB weights).
