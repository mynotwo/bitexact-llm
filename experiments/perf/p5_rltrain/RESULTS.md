# p5_rltrain: C_plugin, the all-integer RL arm through the BitExact vLLM plugin

**Result.** Arm C_plugin does C's sampling and teacher-forced rescoring inside the vLLM plugin, with weights updated in place every step. It matches the standalone integer engine bit for bit and takes 16.8 s/step against 28.2 for C on the same GPU. D takes 14.4. Base: main a0354e1.

## What changed

- `src/bitexact/vllm_plugin/rl.py` (new): `update_weights` copies new int8 constants into the live IntLlama tensors in place, so CUDA graphs see them. It re-runs `post_bound < 2^38` for every linear layer and the lm_head first, and raises OverflowError before writing anything if a bound fails. `int_logp` and `lse_table` equal those in `intq.py`. A hook on vLLM's `PromptLogprobsWorker` returns exact integer log p for tokens P..L-1 of a request carrying `extra_args={"bitexact_score_from": P}` and `skip_reading_prefix_cache=True`, computed on the GPU from the prefill hidden states.
- `src/bitexact/vllm_plugin/sampler.py`: 2 added lines that record the integer log p of sampled tokens while `RL.lp` is on.
- `experiments/abc_rl/plug.py`, `check_plug.py`, and `--arm C_plugin` in `train.py`.

## Identity checks (H800, 64 sequences x 256 tokens per update, step 0 plus 3 weight updates)

| check | differ |
|---|---|
| rollouts, plugin vs `intq.generate` | 0 of 64 sequences at every update |
| sampler log p, plugin vs intq | 0 tokens |
| rescoring, plugin vs `intq.score` | 0 tokens |
| plugin sampler log p vs plugin rescoring | 0 tokens |

Each update changed about 5.6% of the int8 weights. A layer with multipliers inflated 2^20 raised `post_bound 3.3e15 >= 2^38 for 0.wqkv` and left the weights unchanged. In training, mean |sampler - rescoring| was 0.0 over all steps and there were no bound errors.

## Regression (1B suite and pytest, GPU 3, main a0354e1 vs repo)

All 14 suite jobs have identical `.hash` files; `compare.py` shows token_seqs_diff 0 and logit_hash_diff 0 on all 9 short and 4 long cmp lines. Four jobs first produced no output (probably GPU contention) and were rerun alone. pytest: 126 passed, 2 skipped.

## Timing (H800 GPU 0, same session, sequential, 20-step pilots, median of steps 2-19, seed 1)

| arm | s/step | sample | rescoring | trainer |
|---|---|---|---|---|
| A | 11.1 | 5.7 | 0 | 5.2 |
| C (old IntLlama) | 28.2 | 15.1 | 7.7 | 5.3 |
| C_plugin | 16.8 | 5.8 | 5.6 | 5.2 |
| D | 14.4 | 5.7 | 3.3 | 5.2 |

The host was shared (load average about 9). Sampling now equals D and A; the remaining gap to D is rescoring (5.6 vs 3.3 s), which was not profiled. The 200-step runs show a median of 15.7 s/step (seed 1) and 17.2 (seed 2).

## C_plugin, 200 steps x 2 seeds (integer eval, first 500 test questions, greedy)

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| seed 1 | 0.062 | 0.100 | 0.132 | 0.132 | 0.160 | 0.170 | 0.168 | 0.188 | 0.172 |
| seed 2 | 0.062 | 0.080 | 0.102 | 0.112 | 0.138 | 0.082 | 0.120 | 0.124 | 0.132 |

Step-200 accuracy of 0.172 and 0.132 compares with C's 0.177 +- 0.024, range [0.148, 0.204] over 4 seeds. Seed 1 is inside that range; seed 2 is below C's minimum but within A's range [0.056, 0.170] and about 2 SE of a 500-question evaluation. Two seeds cannot separate them. Train reward over the last 20 steps: 0.162 and 0.130, against C's 0.153 +- 0.026.

## Trajectories

At step 0, C and C_plugin have identical rollouts and log p, but different grad norm (0.40480 vs 0.40488), and rollouts diverge from step 1. A rerun of C on the same GPU and seed also differs from the first C run at step 0 (0.40480 vs 0.40491) with identical rollouts. The HF bf16 backward is not bit-reproducible on this host, so trajectories cannot be matched exactly. The old C seed runs used H100; no 20-step C rerun was done there.
