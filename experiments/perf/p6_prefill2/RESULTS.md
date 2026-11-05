# p6_prefill2: C_plugin rescoring (H800, GPU 7 timing, base 25ed90e vs perf6/prefill2)

**Result.** C_plugin's integer rescoring drops from 4.0-4.1 s to 1.5-1.6 s per step (below D's 2.9-3.1 s), and C_plugin goes from 15.0-15.5 to 11.9-12.6 s/step against D 13.4-13.5 and A 10.8-11.3. Same integers: all correctness checks pass. The gain is in the scoring path, not in the prefill forward (the prefill microbench is unchanged).

## What dominated rescoring
Profile of `PlugEngine.score` on the training workload (64 prompts, 2-shot GSM8K, 214 prompt + 214 response tokens on average, 27,378 tokens; 512 rollouts = 8 x 64 for a larger sample, 218,010 tokens):
- Scoring ran the LM head and `rl.int_logp` on int64 logits of every response row (chunks of 1024): ~15 elementwise passes over rows x 128,256 x 8 bytes (`sub`, `clamp`, shifts, two table gathers, adds, sum). In the 512-rollout profile these ATen kernels were about 45% of GPU time (sub 10%, add 7%, index 5.5%, shift 4%, clamp 4%, ...), 3.79 s per score call.
- The scoring request set `skip_reading_prefix_cache=True`, so every request re-prefilled its whole 214-token prompt although the generation step had just cached it (the cache holds the same integers). 
- Scheduler budget (`max_num_batched_tokens` 4096 / 8192 / 16384 / 32768) changed nothing (2.58 / 2.50 / 2.55 / 2.55 s on the 512 set, before the fixes below); host time was not a limit (CPU 2.06 s vs CUDA 2.36 s in the profile, overlapped).

## Changes (all bit-identical)
1. `vllm_plugin/tri_logp.py`: log p from the LM-head GEMM output in two Triton passes (`_stat_kernel`: row max + target logit; `_sum_kernel`: table-interpolated exp sum, int64, exact in any order), then `ln_q` on [n]. No int64 logits are stored. `rl._score_rows` uses it on CUDA with nl = 2, else the old `int_logp`. Test `test_tri_logp_matches_int_logp` compares bits with `rl.int_logp` (M = 1..300, V = 128256 / 5000 / 1023, logit scales 2^18..2^29, targets at max, at min and random).
2. Scoring requests read the prefix cache up to token P-1 (the first scored row) instead of skipping it (`rl.install` wraps `KVCacheManager.get_computed_blocks` for requests with `bitexact_score_from`; `plug.py` drops `skip_reading_prefix_cache`). Rows P-1.. are always computed. Hits come from the blocks generation cached; `PlugEngine.load` still resets the prefix cache after each weight update, so a hit never crosses a weight change.
3. `gemm_skinny.LARGE[(128256, 2048)] = (256, 128, 128, 1, 8, 4)`: Triton tile for the LM head on scored rows (1024 / 2048 / 4096 rows: 0.764 / 1.511 / 3.331 ms cutlass -> 0.635 / 1.239 / 2.537 ms; equal to `_int_mm`, `test_large_equals_int_mm` covers it).

Score time on the 512-rollout set (median of 3-5 calls, same output hash 54cb5916932055f5 in every run): base 3.79 s -> int logp kernel 2.51 s -> + prefix cache 1.60 s -> + LM-head tile 1.59 s. After the change: `_k` 31%, `_pre_kernel` 23%, cutlass `_int_mm` 11%, `_h_kernel` 9%, `_sum_kernel` 8%, `_add_norm` 5%, `_quant` 3%, `_stat_kernel` 3%.

Tried, no gain: attention tile `BITEXACT_PREFILL_BM/BN` on the scoring set (BM 32/BN 64 1.625 s, BM 64/BN 64 1.611 s, BM 16 1.680, BM 32/BN 128 1.706, BM 64/BN 128 1.986): the default stays. Token budget 4096 / 16384 vs 8192: 1.658 / 1.635 vs 1.625 s.

## Timing (H800 GPU 7, one session, sequential, 20-step pilots, seed 1, median of steps 2-19, 2 reps)
| arm | rep | s/step | sample | rescoring | trainer |
|---|---|---|---|---|---|
| A | 1 / 2 | 11.3 / 10.8 | 5.9 / 5.4 | 0 | 5.4 / 5.4 |
| D | 1 / 2 | 13.5 / 13.4 | 5.2 / 5.2 | 3.1 / 2.9 | 5.1 / 5.2 |
| C_plugin, base 25ed90e | 1 / 2 | 15.5 / 15.0 | 5.8 / 5.6 | 4.1 / 4.0 | 5.4 / 5.3 |
| C_plugin, perf6/prefill2 | 1 / 2 | 12.6 / 11.9 | 5.6 / 5.2 | 1.6 / 1.5 | 5.3 / 5.2 |

Launch log (`timing_launch.log`) shows GPU 7 at 0 MiB at every start; load average 3.2-11.7. The prefill microbench was queued behind these runs on the same GPU.

## Prefill microbench (prompt tokens/s, median of 3 interleaved reps)
| B x L | base | branch |
|---|---|---|
| 64 x 256 | 106,663 | 106,073 |
| 64 x 512 | 100,084 | 100,096 |
| 16 x 512 | 99,715 | 99,833 |
| 16 x 1024 | 84,832 | 85,128 |
| 32 x 1024 | 86,063 | 86,003 |
| 64 x 1024 | 83,021 | 83,105 |
Unchanged, as expected: the prefill forward is not modified. Remaining prefill floor is as in p5_prefill (GEMMs ~50%, `_h_kernel` silu epilogue, attention); in scoring, the remaining time is spread over the same kernels (above), with no single item left that is more than a third of it.

## Bit identity (GPU 0 / 3, branch HEAD 88598cc)
- pytest tests/: 133 passed, 2 skipped (4 new tri_logp cases).
- 1B 14-run suite: all 14 `.hash` cmp OK vs base 25ed90e; token_seqs_diff 0, logit_hash_diff 0 on all 9 short + 4 long jobs and the reference.
- Token dumps (mixed temperatures, T=0, logprobs): IDENTICAL.
- `experiments/abc_rl/check_plug.py` (step 0 + 3 updates, 64 sequences): seq_tokens_differ 0, sampler_lp_differ 0, rescore_lp_differ 0, plugin_sampler_vs_rescore_differ 0 at every update; bound check raised as expected; CHECK PASS.
- 8B ref / batch_inv not run: no 8B code path changed (tri_logp is used only by scoring requests; the added `LARGE` key is the 1B LM-head shape; decode, prefill and sampling kernels are untouched). 

## Reproduce
`prof_score.py` + `go.sh TREE GPU NAME [--profile --reps N --budget B]` (scoring profile), `gemm_lm.py` (LM-head tiles), `timing.sh` + `med.py` (pilots), `run.sh` (copy of p5_merge's stage runner pointed at `/data41/audreymao/p6_prefill2`), `check_plug.sh`.
