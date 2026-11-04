# Arm C step time with the packed scorer and the plugin sampler (branch exp/abc-rl-speed)

`exp/abc-rl-speed` = `exp/abc-rl` + merge of `exp/speed`. Arm C (integer) samples with the BitExact vLLM plugin (CUDA graphs, `tri_paged`) and rescores with the packed `bitexact.int8.scorer.Scorer`, selected by `train.py --int-impl new`. `--int-impl old` is the previous path (`intq.generate`, `intq.score`). The RL algorithm, prompts, seeds and hyper-parameters are unchanged. Only `experiments/abc_rl/train.py` and `experiments/abc_rl_speed/` are touched.

## Step time

Host 138, one H100 80GB per run (each started on a GPU at 0 MiB), vLLM 0.32.0, torch 2.13.0+cu130, seed 1, 512 rollouts per step (64 prompts x 8), 256 new tokens, micro-batch 16, no evaluation. The three runs ran at the same time on three GPUs. 30 steps each (steps 0-29 logged); the table is the median of steps 5-29 (25 steps). Medians in seconds.

| arm | s/step | sample | score | train (fwd + bwd) | optim + quantize |
|---|---|---|---|---|---|
| C-new (integer, plugin + packed scorer) | 19.7 | 7.5 | 6.7 | 5.2 | 0.2 |
| C-old (integer, `intq.generate` + `intq.score`) | 30.9 | 16.4 | 8.9 | 5.3 | 0.2 |
| D (vLLM `VLLM_BATCH_INVARIANT=1`, `TRITON_ATTN`) | 11.7 | 4.1 | 2.5 | 4.8 | 0.2 |

C-new is 1.57x faster than C-old (sampling 2.2x, scoring 1.3x) and 1.7x slower than D. C-old reproduces the 30.9 s/step measured earlier on 138. D on the H100 of 138 is 11.7 s/step, against 16.5 s/step on the shared H800 of 141. The per-step range (steps 5-29) is 19.2-20.5 s for C-new, 30.0-34.1 s for C-old, 11.2-12.0 s for D.

The scorer's 5.8x in `experiments/speed/RESULTS.md` is against the eager bench scorer. Against `intq.score` (already batched and compiled) the gain in the RL loop is 1.3x (8.9 s to 6.7 s for about 110k response tokens). Why it is smaller was not profiled.

Settings that differ between the arms, none of which changes any computed value:
- D ran with `--gpu-mem 0.16` (default 0.18) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the first D attempt hit a CUDA OOM in the trainer backward at step 1 on the 79 GiB H100 with the defaults.
- C-new ran with `expandable_segments:True`, `--plugin-gpu-mem 0.20`, `--score-chunk 64`, and drops the scorer's paged K/V cache and calls `empty_cache()` after scoring. Two earlier C-new attempts hit the same OOM (at step 1 and step 12) before these changes. C-old ran with the defaults.
- Other jobs used GPUs 4, 6 and 7 of 138 during the runs.

## Bit check, C-new against C-old

Result: the integer path is bitwise identical between the two code trees for 10 consecutive steps. A bitwise comparison of two RL trajectories is not possible, for a reason independent of this change.

1. Whole RL runs, seed 1. Step 0 is identical: same rewards, tokens, sampler log p and scorer log p (hashes of the float64 values), delta = 0 in all 109,538 tokens, and the loss is bitwise equal (`0x1.fd385dc000000p-9`). The gradient norm already differs in the last digits (0.40487 vs 0.40488), and from step 1 the samples differ. Two runs of the same C-old code (`C_old_det1`, `C_old_det2`, also with `torch.use_deterministic_algorithms`) differ in the same way: step 0 loss equal, gradient norm different, everything after step 0 different. The HF bf16 trainer's backward is therefore not reproducible from run to run, so the weights after the first optimizer step differ between any two runs, and the loss is equal at step 0 only. The loss is not bitwise identical beyond step 0 for this reason; it depends on the sampled tokens, which depend on the weights.
2. Integer path on identical weights (`bitcheck.py`). Both code trees get the same weights at each of 10 steps (fp32 master x (1 + 2e-3 N(0,1)_k), quantised by `intq.quantize`, loaded in place; 5.6-7.9% of the int8 entries differ from the initial ones after step 0), the same prompts and the same seeds as `train.py`. For each step it compares rewards, sampled tokens (512 rollouts, about 110k tokens), sampler log p and scorer log p as lists of float64, and counts sampler-vs-scorer differences in each tree.

| steps | rewards equal | tokens equal | sampler log p equal | scorer log p equal | delta != 0 (old / new) |
|---|---|---|---|---|---|
| 0-9 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 0 / 0 |

Full output: `runs/bitcheck_compare.txt`. The loss is a function of these tokens and log p values and of the shared HF trainer forward (the loss at step 0 above is bitwise equal), so with identical weights it is equal too; it was not recomputed in `bitcheck.py`.

## Files

`train.py` (flags `--int-impl`, `--plugin-gpu-mem`, `--score-chunk`, `--deterministic`; `--eval-every 0` disables evaluation; each step record also logs `loss`, `loss_hex`, `tok_sha`, `slp_sha`, `ilp_sha`), `run_speed.sh` (one timed run), `analyze.py` (table and per-step comparison), `bitcheck.py` / `run_bitcheck.sh` (identical-weights check), `env.sh`, `sync.sh` (copy code to 138: `code_new` = this tree, `code_old` = `exp/abc-rl` plus this `train.py`), `runs/*/steps.jsonl` (per-step records of the runs above; `C_*_det*` are the 10-step determinism runs).
