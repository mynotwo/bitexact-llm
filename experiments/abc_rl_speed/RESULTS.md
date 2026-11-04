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

## Scoring phase profile and int_logp fix

Scoring was not slow because of chunking or memory. `prof_score.py` (one H100, 512 sequences, 211,150 tokens of which 106,382 response tokens, no trainer) gives the same speed at 64 and 512 sequences per `score()` call (33.4k tok/s; chunk 512 only needs 27 GB instead of 14 GB). The bench's 5.8x is against the eager scorer (6.6k tok/s); C-old (`intq.score`, 8.9 s) was never at that baseline, so 1.3x was the gap to a compiled path, and the loop already ran the new scorer at ~32k tok/s (6.7 s). Time split of the 6.3 s: layers + attention about 4.6 s, `int_logp` 1.38 s, lm_head 0.28 s. With float64 log-softmax instead of `int_logp` the same run takes 5.25 s (40.2k tok/s, bench level); `int_logp` made ~10 passes of int64 temporaries over [512, 128256] (peak 14.3 GB vs 10.4 GB).

Fix: `intq.int_logp` computes the vocabulary part (max, clamp, table lookups, interpolation, sum) in one `torch.compile`d function; the value is the same integer arithmetic (`int_logp_ref` keeps the eager version; `test_intlogp.py` checks bit equality on 40 synthetic and real-logit cases). Profile after: 5.05-5.3 s (40-42k tok/s), logp 0.28 s, peak 10.4 GB at 64 sequences per call.

| run (seed 1, 30 steps, median steps 5-29) | s/step | sample | score | train | optim + quantize |
|---|---|---|---|---|---|
| C-new before | 19.7 | 7.5 | 6.7 | 5.2 | 0.2 |
| C-new after (`runs/C_new_v2_s1`) | 17.9 | 6.7 | 5.6 | 5.4 | 0.2 |

Bit check after the change: `bitcheck.py` 10 steps, C-old against C-new, all rewards, tokens, sampler log p and scorer log p equal, delta = 0 (`ALL EQUAL`); also equal to the previous C-new output. Step 0 hashes of the RL run equal the previous run. `scorer.py` unchanged, so the scorer gate is not re-run. Remaining scorer time is layers + attention at bench speed.

## Sampling gap: C-new 6.7 s vs D 4.1 s (profile, no loop change)
`prof_sample.sh` / `prof_sample.py` (one H100 each, sampler built in train.py's order, 512 rollouts = step-0..2 prompts and seeds of the RL run, no trainer; logs in `runs/ps/`). C: `--plugin-gpu-mem 0.20`; D: `--gpu-mem 0.16`; `max_num_seqs 512`.

| | C-new | D |
|---|---|---|
| KV cache (tokens) | 138,048 | 277,200 |
| preemptions per step | 0 | 0 |
| engine steps (1 prefill + 256 decode) | 257 | 257 |
| decode batch, mean / median / max | 430 / 427 / 512 | 420 / 420 / 512 |
| prefill (s) | 0.26 | 0.03-0.58 |
| decode total (s) | 5.9 | 4.3 |
| decode ms/step at that batch | 22.5-23.0 | 16.6-17.7 |
| `generate()` total (s) | 6.2 | 4.5-4.9 |

1. The KV cache is not the limit. All 512 sequences run together (peak batch 512, median 427), nothing is preempted, 138k tokens are enough for 512 x ~450. The 0.20 vs 0.16 split changes nothing here; the plugin KV takes 64 KB/token vs 32 KB/token in vLLM (half the tokens for the same 8.4 GiB).
2. The gap is GPU kernel time per decode step at batch ~430: 22.6 ms measured and 22.6 ms of CUDA time per step in a torch profile (CPU is not on the critical path). Bench B=64 (5.8 ms) scales roughly linearly to ~23 ms at B=430-512. Top CUDA time (share of step): `_acc_kernel` 22%, int8 cutlass GEMM 13%, compiled quantise/requantise triton kernels (`triton_red_fused...amax...`) ~12% + ~12% in two more, `_max_kernel` 9%. D: `matmul_kernel_persistent` 58%, `kernel_unified_attention` 24%, sampling 8%.
3. The sampler's `int_logp` hook costs 0.5 ms/step (22.3 vs 22.8 ms with the hook stubbed, `--nolog`), 0.13 s per step. Not the gap.
4. Loop-side levers (memory split, `max_num_seqs`, trainer offload) have no effect on a run that is already kernel-bound at full batch, so train.py is unchanged and no C-new/D rerun was needed. The remaining 1.5 s is in plugin kernels (src/bitexact/vllm_plugin, `int8`): attention `_acc_kernel`, `_max_kernel`, the requantise triton kernels. Plugin-side note, not changed here.
