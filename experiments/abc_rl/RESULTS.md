# A/B/C GRPO: does the arithmetic behind sampler and trainer log p change RL?

Question 3 of the A/B/C design (`2026-11-03-int-vs-float-train-infer-abc.md`): with the same quantized
weights at every step, train Llama-3.2-1B on GSM8K with GRPO under four arms that differ only in which
arithmetic produces the sampler log p and the trainer log p.

| arm | sampler (rollouts, sampler log p) | trainer log p used in the ratio | gradient |
|---|---|---|---|
| A | vLLM 0.32 bf16 | HF bf16 | float graph |
| A_TIS | vLLM 0.32 bf16 | HF bf16, plus truncated IS weight min(π_train/π_sampler, 2) | float graph |
| B | BitExact int8 engine | HF bf16 | float graph |
| C | BitExact int8 engine | BitExact int8 engine (teacher-forced rescoring) | float graph, logp_float + sg(logp_int − logp_float) |

## Results (2 seeds × 200 steps per arm)

![eval](figs/eval.png)

GSM8K test (first 500, greedy, 2-shot, 256 tokens). Mean over seeds, per-seed values in parentheses.

| arm | integer engine, step 200 | vLLM bf16, step 200 | integer engine, mean of steps 150/175/200 | vLLM bf16, mean of steps 150/175/200 | train reward, last 20 steps |
|---|---|---|---|---|---|
| step 0 (all arms) | 0.062 | 0.050 | | | |
| A | 0.159 (0.154, 0.164) | 0.172 (0.164, 0.180) | 0.151 (0.149, 0.152) | 0.152 (0.151, 0.152) | 0.153 (0.159, 0.147) |
| A_TIS | 0.172 (0.168, 0.176) | 0.168 (0.164, 0.172) | 0.167 (0.171, 0.162) | 0.163 (0.165, 0.160) | 0.147 (0.157, 0.138) |
| B | 0.014 (0.004, 0.024) | 0.013 (0.002, 0.024) | 0.024 (0.010, 0.037) | 0.021 (0.009, 0.032) | 0.027 (0.032, 0.023) |
| C | 0.196 (0.204, 0.188) | 0.193 (0.202, 0.184) | 0.193 (0.204, 0.181) | 0.192 (0.205, 0.179) | 0.173 (0.185, 0.161) |

Findings:

1. **B collapsed in both seeds; no other run did.** B learned like the others until step ~75 (eval 0.098 /
   0.104), then test accuracy fell to 0.01 (seed 1, by step 100) and 0.06 (seed 2, by step 150) and did not
   recover; the float model collapsed too (vLLM eval 0.002 / 0.024), so the weights themselves degraded.
   The share of tokens whose ratio π_float_train/π_int_sampler left [0.8, 1.2] rose from 7e-4 (steps 0–39)
   to 1.9e-2 (seed 1, steps 80–119) before and during the collapse; seed 1's responses ran to the 256-token
   cap and reward dropped to ~0.01. A's ratio-outside-clip share also rose over training (2.5e-3 → ~1e-2) but
   A did not collapse. Mechanism: not established (see Scope).
2. **C ended highest**: 0.196 vs A 0.159 and A_TIS 0.172 (integer-engine eval; vLLM eval 0.193 vs 0.172 /
   0.168). The seed ranges do not overlap with A, but with 2 seeds and a binomial SE of ~0.017 per
   500-question eval this is suggestive, not established.
3. **A vs A_TIS: difference not detected** (within seed noise).
4. In C the sampler/trainer gap is exactly 0 on every token (integer rescoring reproduces the sampler's
   integer logits). The float trainer's own log p differs from the integer one by a mean 0.022 at step 0,
   shrinking to 0.012 by step 200.
5. Integer and float evaluations of the same weights track each other within ~0.02 in every run.

![mismatch](figs/mismatch.png)
![reward](figs/reward.png)
![stability](figs/stability.png)

Sampler vs trainer mismatch over all 400 steps per arm (Δ = log p_trainer − log p_sampler on sampled tokens):

| arm | mean \|Δ\| | max \|Δ\| | tokens Δ≠0 | ratio outside [0.8,1.2] | PPO clip-active | KL k3 | grad norm max | non-finite grads | bound errors |
|---|---|---|---|---|---|---|---|---|---|
| A | 1.94e-02 | 4.17 | 0.992 | 6.3e-03 | 1.0e-03 | 9.7e-04 | 1.53 | 0 | 0 |
| A_TIS | 1.91e-02 | 5.22 | 0.996 | 5.2e-03 | 0 | 8.9e-04 | 1.27 | 0 | 0 |
| B | 1.94e-02 | 5.13 | 1.000 | 7.5e-03 | 4.7e-04 | 1.1e-03 | 2.10 | 0 | 0 |
| C | 0 | 0 | 0 | 0 | 0 | 0 | 1.08 | 0 | 0 |

(A_TIS: the PPO ratio is π_train/sg(π_train) = 1 and the mismatch enters only through the truncated weight,
so its clip-active share is 0 by construction.)

## Can the int8 constants be rebuilt every step?

Yes. Each step quantizes the fp32 master to a fresh int8 constants set: per-row int8 weights, multipliers
m_c and the per-matrix exponent fm are recomputed by the BitExact builder's rule
(fm = min(60, ⌊62 − log2(worst)⌋ − 1), which is what keeps |acc·m| < 2^62 for any activation within limbs);
RMSNorm gammas are rounded at 12 bits; tables (exp, SiLU, RoPE, ln) do not depend on weights and are reused.

- The per-step quantizer reproduces the original constants file bit for bit at step 0 (`check_setup.py`,
  CHECK1). A first version computed the scales on the GPU, where CUDA turns w / s into w · (1/s): ~1e-4 of
  the weights rounded differently. Scales and multipliers are now computed on the CPU.
- fm stayed in 29..32 for all 800 training steps. 0.2% of int8 entries changed at step 1 and 0.03–0.04% per
  step at the end (B seed 2 after collapse: 0.006%) at lr 1e-6.
- The engine's device-side bound checks (`model.check()`) passed for every integer forward of all B and C
  steps and every integer-engine evaluation (including those of A runs): 0 bound errors. These are the
  runtime guards of the engine, not a static proof for all inputs.

## Setup

- Model Llama-3.2-1B (base, unsloth mirror). fp32 master weights (tied embedding), AdamW (β 0.9/0.999,
  wd 0), lr 1e-6 constant, grad-norm clip 1.0.
- GRPO: 64 GSM8K-train prompts per step × G = 8, T = 1, no top-k/top-p, ≤ 256 new tokens, stop at EOS. 2-shot
  prompt (train examples 0 and 1, excluded from training). Reward 1 iff `#### N` in the text before the first
  `\nQuestion:` equals the gold answer. Advantages normalized per group ((r − mean)/(std + 1e-4)). Token-mean
  loss over the 512 responses, one optimizer step per RL step (gradient accumulation over microbatches of
  16). PPO clip ε = 0.2. No reference-model KL (β = 0); the logged KL is sampler vs trainer.
- Same prompts and request seeds per step across arms for a given seed (seed s: prompt order from
  `random.Random(s)`).
- Weights every step: int8 quantization of the master. Float arms load the dequantized values
  (q · m_c · 127 / 2^fm, gammas and embedding at the engine's fixed point; lm_head = dequantized int8
  embedding, untied) in bf16 into both vLLM (in-process `load_weights`, prefix cache reset) and the HF
  trainer. Straight-through gradient: master grad = grad w.r.t. the dequantized weights (embedding and
  lm_head grads summed for the tied master).
- Integer sampler: standalone `bitexact.int8` engine (batched prefill + KV-cache decode, Gumbel-max keyed by
  (request seed, t) with the vLLM plugin's GPU PCG64). It matches the plugin: 16/16 rollouts of the
  rl_mismatch plugin run reproduced token for token (CHECK3).
- Integer log p (`intq.int_logp`): integer log-softmax of the int64 logits row (44-bit exp table with
  rounded interpolation, the engine's ln table); within 5e-9 of a float64 log-softmax and batch-invariant.
- Eval every 25 steps and at the end: GSM8K test first 500, greedy, both the integer engine and vLLM bf16
  with the current weights.
- Hardware: 8× H100 80GB (host 138, driver 595), torch 2.13 + cu130, vLLM 0.32.0, transformers 5.19. One
  run per GPU; GPUs were shared with other users' jobs, so timings are under load.

## Cost (pilot: 10 steps per arm, 1 GPU each; full runs agree)

| arm | s/step | sample | integer rescoring | trainer fwd+bwd | quantize + weight load | peak torch memory |
|---|---|---|---|---|---|---|
| A | 10.3 | 5.1 | – | 5.1 | 0.13 | 71.5 GB |
| A_TIS | 10.0 | 4.8 | – | 5.0 | 0.13 | 71.5 GB |
| B | 21.4 | 16.2 | – | 5.1 | 0.12 | 71.5 GB |
| C | 29.8 | 16.2 | 8.3 | 5.1 | 0.12 | 71.5 GB |

Full runs (median s/step): A 11.0, A_TIS 10.3, B 21.4, C 30.8. A 200-step run including 9 evaluations took ~40 min (A),
~75 min (B) and ~1 h 45 min (C). Peak memory includes vLLM's in-process reservation (gpu_memory_utilization 0.18) and is
the same for all arms. Evaluation of 500 greedy questions: integer engine ~14 s, vLLM ~2.8 s.

## Deviations from the design

- The integer sampler is the standalone engine, not the vLLM plugin: the plugin loads constants at start-up,
  while RL swaps weights every step. Token identity with the plugin was checked (above).
- The original-bf16-model A reference row was not run.
- No reference-model KL (allowed by the task: "optional/small").
- Two launches failed for lack of GPU memory taken by other users' jobs (C seed 1 on GPU 0, B seed 1 on GPU 3)
  and were restarted from scratch on free GPUs; logs kept as `*_aborted.log`, `*_oom.log`, `*_nomem.log` on 138.

## Scope

- One model, one task, 200 steps, 2 seeds per arm. Accuracy differences of ~0.03 are about two binomial SEs of
  one evaluation; treat C > A as suggestive.
- B's collapse is observed in 2/2 seeds, and 0/6 runs of the other arms. Why B diverges while A, whose
  sampler/trainer gap is of similar size, does not, was not tested. A plausible but unverified explanation:
  in B the gap is systematic (integer activations vs float activations), so π_int and π_float are different
  policies and the float update is applied off-policy; B + TIS would test this.
- The PPO ratio here only measures arithmetic mismatch (one optimizer step per batch), so these results are
  about mismatch, not about multi-epoch PPO.
- The float arms use the dequantized int8 weights in bf16; results for a stock bf16 model may differ.

## Reproduce

```
# on a GPU host; env: PY (vLLM 0.32 venv with this repo on PYTHONPATH), HF (Llama-3.2-1B dir),
# CONST (int8 constants from examples/build_constants.py), DATA (GSM8K train/test jsonl), C (this dir)
$PY $C/check_setup.py --hf $HF --const $CONST --data $DATA --rollout PLUGIN_ROLLOUT.json
OUT=runs bash $C/run_full.sh "0:C_s1" "1:C_s2" "3:B_s1" "5:B_s2" "6:A_s1,A_TIS_s1" "7:A_s2,A_TIS_s2"
python $C/analyze.py --runs runs --out figs
```

Files: `intq.py` (per-step quantization, integer engine wrapper, integer log p, batched integer sampling and
scoring), `floatm.py` (vLLM sampler with weight sync, HF trainer with straight-through), `train.py` (GRPO),
`check_setup.py`, `run_full.sh`, `analyze.py`.
