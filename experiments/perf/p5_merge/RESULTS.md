# p5_merge: perf5/prefill + perf5/rltrain on main a0354e1

**Result.** The two lanes compose. All correctness checks pass on the merged tree (fused prefill on), and the fused prefill cuts C_plugin's integer rescoring from 5.2-5.3 s to 3.8-4.0 s, so C_plugin goes from 16.1-16.5 to 14.3-15.1 s/step, against 13.2-14.0 for D (same GPU, same session). Rescoring is still about 1 s above D's; sampling and trainer time are equal across arms. No merge fixes were needed.

## Timing (H800 GPU 7, one session, sequential, 20-step pilots, seed 1, median of steps 2-19, 2 reps)

| arm | rep | s/step | sample | rescoring | trainer |
|---|---|---|---|---|---|
| A | 1 / 2 | 11.4 / 10.3 | 5.9 / 5.2 | 0 | 5.5 / 5.2 |
| D | 1 / 2 | 14.0 / 13.2 | 5.4 / 5.1 | 3.2 / 2.9 | 5.3 / 5.2 |
| C_plugin, BITEXACT_PREFILL_FUSE=0 | 1 / 2 | 16.5 / 16.1 | 5.7 / 5.6 | 5.3 / 5.2 | 5.4 / 5.3 |
| C_plugin, fused prefill | 1 / 2 | 15.1 / 14.3 | 5.7 / 5.3 | 4.0 / 3.8 | 5.4 / 5.2 |

Fused prefill saves 1.4 s (rep 1) and 1.8 s (rep 2) per step, all of it in rescoring (-1.3 / -1.4 s). Run-to-run spread of one arm is about 0.8 s (rep 1 is slower than rep 2 for every arm), so compare within a rep. GPU 7 showed 0 MiB before each launch (log: `timing_launch.log` in the 141 dir `/data41/audreymao/p5_merge/out`, load average 1.6-2.2 at all launches except rep 2 A, 9.3). `trainer` = quantize + backward + optim.

## Correctness (GPUs 0 and 3, merged tree 0e6117a, fused prefill on by default)

- pytest tests/: 129 passed, 2 skipped.
- 1B 14-run suite: all 14 `.hash` files cmp-identical to a0354e1; token_seqs_diff 0 and logit_hash_diff 0 on all 9 short and 4 long jobs, plus the reference.
- Token dumps (mixed temperatures, T=0, logprobs): IDENTICAL to a0354e1.
- 8B plugin vs IntLlama reference (7 runs): compare_exit 0, token_seqs_diff 0; 8B batch_inv PASS (graph and eager, 192 rows).
- check_plug.py, step 0 plus 3 weight updates, 64 sequences per update: seq_tokens_differ 0, sampler_lp_differ 0, rescore_lp_differ 0, plugin_sampler_vs_rescore_differ 0 at every update; int8 changed fraction 0.0 / 0.0562 / 0.0560 / 0.0559, equal for intq and plugin; bound check raised OverflowError as expected and CHECK PASS.

check_plug ran on GPU 0 at the same time as the 8B jobs (correctness only, no timing).

## Reproduce
`timing.sh` (GPU 7 pilots), `med.py` (medians), `check_plug.sh`, `run.sh` (p5_prefill's run.sh pointed at this lane's dirs; stages `pytest suite suitecmp tokens tokcmp s8 bi8` with GPU 3 and 0). Base hashes/tokens come from the earlier a0354e1 runs in the p5_rltrain and p5_prefill 141 dirs.
