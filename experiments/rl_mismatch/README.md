# Rollout/trainer log-prob mismatch and failure replay

Two questions, measured on Llama-3.2-1B and GSM8K:

1. In GRPO-style RL, the sampler (vLLM) and the trainer (a separate forward pass) are supposed to
   compute the same token probabilities. How far apart are they with a standard stack, and with BitExact?
2. When a grader flags a failed trajectory, can the same prompt and seed reproduce it token for token
   (needed for oversight, debugging and red-team findings)?

Inference only. No training was run, so this does not measure whether removing the mismatch improves RL.

## Results

Mismatch between sampler-side and scorer-side log p over the sampler's tokens
(Δ = log p_scorer − log p_sampler, r = exp Δ, clip = [0.8, 1.2] as in PPO/GRPO with ε = 0.2):

| sampler → scorer | tokens | mean \|Δ\| | max \|Δ\| | tokens with Δ ≠ 0 | token r outside clip | argmax differs | sequence r outside clip |
|---|---|---|---|---|---|---|---|
| vLLM bf16 (H800) → HF bf16, batch 8 | 216,083 | 0.0251 | 0.660 | 99.99% | 0.22% | 1.48% | 74.4% |
| vLLM bf16 → HF bf16, batch 1 | 216,083 | 0.0251 | 0.660 | 99.99% | 0.22% | 1.48% | 74.2% |
| vLLM bf16 → HF fp32, batch 8 (TF32 off) | 216,083 | 0.0217 | 0.974 | 99.99% | 0.02% | 2.09% | 63.6% |
| HF bf16 batch 1 → HF bf16 batch 8 | 216,083 | 8.1e-6 | 0.165 | 0.039% | 0 | 0.0009% | 0.20% |
| BitExact plugin (H800) → BitExact H800, batch 8 | 215,899 | 0 | 0 | 0 | 0 | 0 | 0 |
| BitExact plugin (H800) → BitExact H800, batch 32 | 215,899 | 0 | 0 | 0 | 0 | 0 | 0 |
| BitExact plugin (H800) → BitExact H100, batch 16 | 215,899 | 0 | 0 | 0 | 0 | 0 | 0 |
| BitExact plugin (H800) → BitExact x86 CPU, batch 4 | 2,954 (15 seqs) | 0 | 0 | 0 | 0 | 0 | 0 |

For the BitExact rows the SHA-256 of the int64 logits row is also identical at every position.

Replay of grader-flagged failures (same prompts and seeds, re-sampled):

| reference → replay | flagged replayed token for token | median first divergence (token) |
|---|---|---|
| vLLM bf16 → same config, H800 | 176 / 1,014 (17.4%) | 64 |
| vLLM bf16 → shuffled order, max_num_seqs 32 | 136 / 1,014 (13.4%) | 50 |
| vLLM bf16 → max_num_seqs 8 | 140 / 1,014 (13.8%) | 49 |
| vLLM bf16 → same config, H100 | 162 / 1,014 (16.0%) | 62 |
| BitExact plugin → shuffled order, max_num_seqs 32, H800 | 1,007 / 1,007 | — |
| BitExact plugin → same config, H100 | 1,007 / 1,007 | — |
| BitExact plugin → standalone engine, batch 64, H800 | 1,007 / 1,007 | — |
| BitExact plugin → standalone engine, batch 16, H100 | 1,007 / 1,007 | — |
| BitExact plugin → standalone engine, x86 CPU (subset) | 15 / 15 | — |

With stock vLLM, 4 to 10 flagged trajectories were graded correct on replay. With BitExact no verdict changed.

## Setup

- Prompts: the first 128 GSM8K test questions, 2-shot prompt from GSM8K train examples 0 and 1.
  G = 8 samples per prompt (1,024 rollouts), request seed 1 + 8p + g, T = 1, no top-k/top-p,
  at most 256 new tokens.
- Baseline: stock vLLM 0.32.0 with bf16 weights; sampler log p = `raw_logprobs`. Scorer = HF transformers
  (sdpa), teacher forcing on prompt + sampled tokens, right-padded batches.
- BitExact: the int8 vLLM plugin samples from integer logits; sampler log p is a float64 log-softmax of the
  sampled int64 logits row (`common.int_logp`). The scorer is the standalone `bitexact.int8` engine with
  the same `int_logp`.
- Grader: rule-based; the response is cut at the first `\nQuestion:` and gets reward 1 iff it contains
  `#### N` equal to the gold answer. Flagged = reward 0 (wrong answer or no answer).
- Hardware: 8×H800 (driver 535), 8×H100 (driver 595), Xeon Platinum 8457C. GPUs were shared with other jobs.
- The plan (prompts, engines, metrics, replay configs) was fixed before results were read.
  Changes made later: H100 standalone replay used batch 16 instead of 64, and a plugin rollout on H100
  plus a standalone batch-64 replay on H800 were added.

## Scope

- One model, one task, 256-token responses. Larger models and longer contexts were not tested.
- BitExact runs an int8 model, which differs from the bf16 model by about 0.09 in logits (see the main
  README). Accuracy of the two runs is therefore not comparable; this experiment only compares each
  engine's sampler with its own scorer.
- Logits match bit for bit, but log p computed from them also needs the same float64 log-softmax code on
  both sides. With numpy 1.26 on the scorer and numpy 2.3.5 on the sampler, 6.5% of log p values differed
  by at most 7.1e-15. The table above uses numpy 2.3.5 on both sides. Comparing integer logits directly
  avoids this.
- The baseline divergence was not attributed to individual kernels.

## Reproduce

```
# 8 GPUs with ~25 GB free each (H800 here); see the header of each script for the variables.
R=/path/to/workdir PY=python BF16=bf16_model_dir INT=intmodel_dir CONST=int8_const.pt \
  DATA=gsm8k_jsonl_dir G0=0 G1=1 G2=2 G3=3 bash run_141.sh
# second GPU type and CPU; needs $R/out/b0_bx.json and $R/out/cpu_seeds.txt from the first host
R=/path/to/workdir PY=python CONST=int8_const.pt TOK=tokenizer_dir DATA=gsm8k_jsonl_dir GPU=0 \
  [VPY=vllm_python INT=intmodel_dir] bash run_138.sh
bash analyze_all.sh
```

A full run takes about an hour. Results were produced at commit 105aac8 (base 38ca4ef); later commits
on main do not touch the modules these scripts import (`intops`, `vllm_plugin`, `llama`, `generate`).
