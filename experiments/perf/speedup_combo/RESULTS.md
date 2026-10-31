# perf/combo (fuse-elementwise + sampler-sync): verification and throughput

## Conclusion

1. **Bit-identity: PASS.** The full plugin invariance suite (`tests/vllm/run_tests.sh`, 14 runs) was run for the combo tree and for 48a60f7 on the same GPU. Every combo run compared with the matching baseline run gives token_seqs_diff 0, missing 0 and logit_hash_diff 0. All 14 combo hash files are byte-identical to the baseline files. The production path (no hash log) gives the same output tokens as the baseline and as perf/sampler-sync: B=32/128, T=1 and mixed T=0/0.7/1.0, 4/4 digests equal.
2. **Throughput at T=0** (bench, second pass, GPU 0, same session as the baseline) rises **1.65x / 1.86x / 2.29x** at B=1/32/128: 225 → 372, 4,438 → 8,272 and 8,454 → 19,354 tok/s. That is more than the product of the two single-branch speedups (B=128: 1.42 × 1.33 ≈ 1.89). Once the sampler no longer blocks the host, the GPU runs back to back, so the shorter elementwise kernels shorten the step directly instead of being hidden behind the idle gap.
3. **Against vLLM bf16 with `VLLM_BATCH_INVARIANT=1` and `attention_backend=TRITON_ATTN`** (the bf16 setting that is batch-invariant on this model, see det_compare) on the same GPU, combo is **1.77x faster at B=1 and 1.11x faster at B=32, but reaches 0.55x of its throughput at B=128** (19,354 vs 34,941 tok/s).

## Setup

- Branch `perf/combo` = `perf/fuse-elementwise` (e35dc2c, code 1295e53) merged with `perf/sampler-sync` (b58ede5, code 277e091) in merge 6abfa58. The merge was clean (the two branches touch disjoint files). Runner: `experiments/perf/combo/run.sh` (85a8651). Exported with `git archive` to `/data41/audreymao/speedup_20261030/combo/{repo,base}`, baseline 48a60f7.
- Suite: server 141, **GPU 1**, 07:06–07:29 server clock (`STAGES=suite`), outputs `/data41/audreymao/speedup_20261030/combo/out/suite/`.
- Bench and BI bench: **GPU 0**, 07:46–07:54 (`STAGES="bench bi"`), outputs `.../combo/out/{bench,bi}/`. The perf/sampler-sync bench ran on the same GPU at 07:25–07:31 (`.../sync/out_v2/bench/`). Every launch waited for 0 MiB used and no compute process.
- Production-path token digests: `experiments/perf/sync/tok_digest.py` (from perf/sampler-sync), GPU 0, 07:54–08:02, outputs `.../combo/out/digest/`.
- Bench: `tests/vllm/bench.py` (integer plugin) and `experiments/perf/det_compare/bench_attn.py` (bf16, `--attn TRITON_ATTN`). Both use 128 random input tokens and 128 output tokens per request, `ignore_eos`, greedy, CUDA graph, `max_num_seqs=B`, and time the second of two identical `generate` calls. bf16 model: `/data41/audreymao/perf_20261030/bf16model` (the same weights the integer model was built from).

## Bit-identity

| check | result |
|---|---|
| suite, combo vs its own `run_ref` (10 short runs, 12 targets × 48 positions; 4 long runs, 3 targets × 16 positions; T = 0 / 0.7 / 1.0) | all 0; compare_exit 0, compare_long_exit 0; no Traceback |
| suite, baseline 48a60f7 (same GPU, same session) | same, all 0 |
| each combo run vs the matching baseline run (`compare.py`, baseline seed-0 warm-up rows dropped), 14 pairs | 14/14: token_seqs_diff 0, missing 0, logit_hash_diff 0, inconsistent_repeats 0 |
| `cmp` of `repo_X.hash` vs `base_X.hash`, all 14 X | 14/14 byte-identical |
| production path, `tok_digest.py`, baseline vs sampler-sync vs combo, both passes | B=32 T=1 `fc5557fb…`, B=32 mixed `fd23421f…`, B=128 T=1 `f39b2fe4…`, B=128 mixed `9775aff2…`: identical across all three trees |

## Throughput (output tok/s, bench second pass, T=0, GPU 0)

| B | 48a60f7 | sampler-sync (277e091) | combo (6abfa58) | vLLM bf16 BI + TRITON_ATTN | combo / base | combo / BI |
|---|---|---|---|---|---|---|
| 1 | 225.4 | 317.8 | 371.7 | 209.7 | 1.65x | 1.77x |
| 32 | 4,438.4 | 5,718.0 | 8,271.7 | 7,449.4 | 1.86x | 1.11x |
| 128 | 8,453.9 | 12,064.7 | 19,353.7 | 34,940.6 | 2.29x | 0.55x |

- The baseline and combo columns come from the same session, with the two trees alternating per cell. The sampler-sync column ran on the same GPU 15–25 minutes earlier. Its own baseline was 225.1 / 4,419.0 / 9,039.3, so the B=128 baseline varies by about 7% between sessions. Against the higher baseline, combo is 2.14x at B=128.
- For reference (not this session): perf/fuse-elementwise alone measured 240.4 / 5,725.2 / 13,247.1 on GPU 0 earlier the same day (`experiments/perf/speedup_fuse/RESULTS.md`).
- Each value is recomputed as out_tokens / sec from its RESULT line (e.g. 16,384 / 0.847 s = 19,344; the gap to 19,354 comes from the rounding of `sec`).

## Caveats

- One timed pass per cell. Other users' jobs ran on GPUs 2–7, and CPU contention was not measured.
- Only T=0 throughput was measured for combo. At T=1 only token identity was checked (digests).
- No profiler run on combo. The explanation in Conclusion 2 (the idle gap removal exposes the elementwise savings) is inferred from the two single-branch profiles and was **not verified** with a combo profile.
- The BI + TRITON_ATTN comparison is a throughput reference only. That setting is batch-invariant on this GPU and vLLM version. It does not give the cross-hardware integer bit-identity of the plugin.
- The caveats of the two component branches still apply (sampler-sync: the overflow flag is read a step or two late, and speculative decoding is rejected by an assert).
