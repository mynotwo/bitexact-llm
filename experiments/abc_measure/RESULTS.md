# A/B/C measurement: sampler/scorer log-prob mismatch, length, speed

Llama-3.2-1B, H800 (GPUs 4 and 5 of host 141), vLLM 0.32.0, torch 2.13, transformers 5.19, numpy 2.3.5 on both sides.
Design: `Obsidian Vault/Claude/2026-11-03-int-vs-float-train-infer-abc.md` (Q1 mismatch, Q2 length, Q4 speed). Inference only; no RL training here (Q3 is the other branch).

| arm | sampler | scorer | weights |
|---|---|---|---|
| A_orig | vLLM bf16 | HF bf16 / HF fp32 | original bf16 checkpoint |
| A | vLLM bf16 | HF bf16 / HF fp32 | BitExact int8 weights, dequantized |
| B | BitExact vLLM plugin (integer) | HF bf16 / HF fp32 | same dequantized weights |
| C | BitExact vLLM plugin | BitExact standalone int8 engine | int8 constants |

d = log p_scorer - log p_sampler over the sampler's tokens, r = exp(d), clip = [0.8, 1.2]. "Sequence r" uses the sum of d over the sequence (cumulative log-ratio S).

## Setup

- Same-weights checkpoint (`export_dequant.py`): weight = q * s per output row (q int8, s from the engine's integer multiplier), gammas = integer / 2^12, embedding = integer / 2^16; the tied head becomes a separate quantized `lm_head`. Written as fp32 safetensors.
  **bf16 is not exact:** only 1.4% of the 1.24e9 quantized weights are representable in bf16; 75.1% are exact in fp32, the rest round with relative error <= 6.0e-8. The checkpoint is therefore fp32. The dequantized weights differ from the original bf16 weights by at most 0.54 of a row scale s (int8 rounding plus the multiplier rounding), i.e. the checkpoint is the int8 model, not the original model.
- Q1 workload as in `rl_mismatch`: first 128 GSM8K test questions, 2-shot prefix, G = 8, request seed 1 + 8p + g, T = 1, no top-k/top-p, <= 256 new tokens, stop at EOS. HF scoring: sdpa, teacher forcing, right-padded batches of 8, TF32 off, float32 log-softmax. Engine scoring: standalone int8 engine, float64 log-softmax of the int64 logits row (`common.int_logp`).
- Q2 workload: prompts 0..N-1, G = 1, `ignore_eos`, fixed length 256 / 1024 / 4096 / 16384 with N = 64 / 32 / 16 / 8 sequences. The 2-shot GSM8K prompt makes the base model keep writing new Question/Answer pairs, so the long text is repeated-format math word problems, not one long chain of thought. HF batch 8 at 256, batch 1 above; engine batch 1; sequences are prefilled in 256-token chunks.
- Q4 on one GPU (GPU 5), 3 timed reps after one untimed warm-up, median.
- Code: `experiments/abc_measure/` (`rollout.py`, `score.py`, `analyze_len.py`, `speed_*.py`, `run_main.sh`, `run_q*_analysis.sh`, `gen_tables.py`), reusing `experiments/rl_mismatch/` (`common.py`, `analyze.py`, `bitexact_engine.py`). Raw metric/speed JSON in `results/`; raw rollouts and scores stay on 141 in `/data41/audreymao/abc_measure/out`.

### Q1

| arm (sampler -> scorer) | tokens | mean abs d | max abs d | tokens d!=0 | token r outside clip | argmax differs | sequence r outside clip |
|---|---|---|---|---|---|---|---|
| A_orig: vLLM bf16 (orig) -> HF bf16 | 216,416 | 0.0252 | 0.792 | 99.99% | 0.23% | 1.49% | 74.71% |
| A_orig: vLLM bf16 (orig) -> HF fp32 | 216,416 | 0.0219 | 0.613 | 99.99% | 0.02% | 2.10% | 63.18% |
| A: vLLM bf16 -> HF bf16 | 214,914 | 0.0253 | 0.44 | 99.99% | 0.22% | 1.51% | 72.75% |
| A: vLLM bf16 -> HF fp32 | 214,914 | 0.0227 | 0.41 | 99.99% | 0.03% | 2.09% | 64.84% |
| B: plugin -> HF bf16 | 215,899 | 0.0241 | 0.562 | 100.00% | 0.07% | 2.20% | 66.60% |
| B: plugin -> HF fp32 | 215,899 | 0.00121 | 0.0495 | 100.00% | 0 | 0.08% | 0 |
| C: plugin -> BitExact engine | 215,899 | 0 | 0 | 0 | 0 | 0 | 0 |

### Q2 per-position bucket, mean abs d (tokens in bucket)

| run | 0-256 | 256-1k | 1k-4k | 4k-16k |
|---|---|---|---|---|
| 256-token gens (64 seqs), A, HF bf16 | 0.0258 |  |  |  |
| 256-token gens (64 seqs), A, HF fp32 | 0.0226 |  |  |  |
| 256-token gens (64 seqs), B, HF bf16 | 0.0246 |  |  |  |
| 256-token gens (64 seqs), B, HF fp32 | 0.00127 |  |  |  |
| 256-token gens (64 seqs), C | 0 |  |  |  |
| 1024-token gens (32 seqs), A, HF bf16 | 0.0259 | 0.0248 |  |  |
| 1024-token gens (32 seqs), A, HF fp32 | 0.0225 | 0.0233 |  |  |
| 1024-token gens (32 seqs), B, HF bf16 | 0.0248 | 0.026 |  |  |
| 1024-token gens (32 seqs), B, HF fp32 | 0.00124 | 0.00122 |  |  |
| 1024-token gens (32 seqs), C | 0 | 0 |  |  |
| 4096-token gens (16 seqs), A, HF bf16 | 0.0258 | 0.0251 | 0.0251 |  |
| 4096-token gens (16 seqs), A, HF fp32 | 0.0229 | 0.0237 | 0.0235 |  |
| 4096-token gens (16 seqs), B, HF bf16 | 0.0246 | 0.0263 | 0.0253 |  |
| 4096-token gens (16 seqs), B, HF fp32 | 0.00123 | 0.00126 | 0.00118 |  |
| 4096-token gens (16 seqs), C | 0 | 0 | 0 |  |
| 16384-token gens (8 seqs), A, HF bf16 | 0.0255 | 0.0226 | 0.0229 | 0.0254 |
| 16384-token gens (8 seqs), A, HF fp32 | 0.0234 | 0.0215 | 0.0225 | 0.0232 |
| 16384-token gens (8 seqs), B, HF bf16 | 0.0234 | 0.026 | 0.0252 | 0.0265 |
| 16384-token gens (8 seqs), B, HF fp32 | 0.0012 | 0.00125 | 0.00116 | 0.00135 |
| 16384-token gens (8 seqs), C | 0 | 0 | 0 | 0 |

### Q2 cumulative sequence log-ratio S_T (sum of d over first T tokens): mean abs S / mean S / fraction of sequences with exp(S) outside [0.8,1.2]

| gen length (seqs) | arm | T=256 | T=1k | T=4k | T=16k |
|---|---|---|---|---|---|
| 256 (64) | A, HF bf16 | 0.675 / -0.219 / 75.00% |  |  |  |
| 256 (64) | A, HF fp32 | 0.506 / -0.149 / 75.00% |  |  |  |
| 256 (64) | B, HF bf16 | 0.504 / -0.216 / 79.69% |  |  |  |
| 256 (64) | B, HF fp32 | 0.03 / 0.00298 / 0 |  |  |  |
| 256 (64) | C | 0 / 0 / 0 |  |  |  |
| 1024 (32) | A, HF bf16 | 0.53 / -0.077 / 62.50% | 1.18 / -0.817 / 84.38% |  |  |
| 1024 (32) | A, HF fp32 | 0.463 / -0.0402 / 68.75% | 0.997 / -0.682 / 87.50% |  |  |
| 1024 (32) | B, HF bf16 | 0.512 / -0.223 / 78.12% | 1.07 / -0.768 / 84.38% |  |  |
| 1024 (32) | B, HF fp32 | 0.0327 / 0.00358 / 0 | 0.0541 / -0.00233 / 0 |  |  |
| 1024 (32) | C | 0 / 0 / 0 | 0 / 0 / 0 |  |  |
| 4096 (16) | A, HF bf16 | 0.679 / -0.0529 / 81.25% | 1.37 / -0.896 / 81.25% | 3.74 / -3.74 / 100.00% |  |
| 4096 (16) | A, HF fp32 | 0.581 / -0.0808 / 87.50% | 1.09 / -0.697 / 87.50% | 2.72 / -2.71 / 93.75% |  |
| 4096 (16) | B, HF bf16 | 0.44 / -0.0282 / 68.75% | 0.856 / -0.553 / 81.25% | 2.77 / -2.67 / 100.00% |  |
| 4096 (16) | B, HF fp32 | 0.0314 / 0.00887 / 0 | 0.0517 / -0.00298 / 0 | 0.0998 / -0.0215 / 0 |  |
| 4096 (16) | C | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |  |
| 16384 (8) | A, HF bf16 | 0.415 / -0.179 / 75.00% | 1.31 / -1.08 / 100.00% | 2.57 / -2.57 / 100.00% | 12.8 / -12.8 / 100.00% |
| 16384 (8) | A, HF fp32 | 0.454 / 0.0184 / 87.50% | 0.791 / -0.432 / 100.00% | 1.16 / -1.04 / 87.50% | 7.45 / -7.45 / 100.00% |
| 16384 (8) | B, HF bf16 | 0.404 / 0.143 / 75.00% | 0.765 / -0.298 / 62.50% | 2.37 / -2.22 / 100.00% | 10.3 / -10.3 / 100.00% |
| 16384 (8) | B, HF fp32 | 0.0337 / -0.000347 / 0 | 0.0652 / 0.00256 / 0 | 0.0954 / -0.00238 / 0 | 0.185 / -0.0104 / 37.50% |
| 16384 (8) | C | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

### Q4 sampler tok/s (median of 3)

| gen len | batch | vLLM bf16 | BitExact plugin | ratio |
|---|---|---|---|---|
| 256 | 1 | 667 | 215 | 3.1x |
| 256 | 8 | 4,996 | 1,263 | 4.0x |
| 256 | 64 | 23,481 | 6,500 | 3.6x |
| 4096 | 1 | 660 | 224 | 2.9x |
| 4096 | 8 | 4,648 | 1,094 | 4.2x |
| 4096 | 64 | 19,136 | 4,019 | 4.8x |

### Q4 scorer tok/s (median of 3; tokens = prompt + response tokens pushed through the network)

| workload | HF bf16 | HF fp32 | BitExact engine |
|---|---|---|---|
| 1024 GSM8K rollouts (215,899 response tokens), batch 8 | 121,576 | 17,006 | 6,501 |
| 16 x 4096-token sequences, batch 4 | 93,833 | 12,831 | 3,552 |

## Findings

1. **Q1.** C is exactly 0 (all 215,899 logits rows hash-identical to what the plugin sampled from, log p equal bit for bit). A and B with a bf16 scorer are the same size: mean |d| 0.0253 (A) vs 0.0241 (B), 66-73% of sequences outside the clip. B with an fp32 scorer is 20x smaller than A with an fp32 scorer (0.0012 vs 0.0227), no token and no sequence outside the clip, argmax differs on 0.08% of tokens (A: 2.1%). So an fp32 trainer already removes almost all of the mismatch for the integer sampler, but not for vLLM bf16.
2. **Q2.** Per-token |d| does not grow with position for any arm (A bf16 0.0255 / 0.0226 / 0.0229 / 0.0254 over the four position buckets of the 16k run; B fp32 0.0012-0.0014). Sequence drift grows with length: A bf16 mean S = -0.22 / -0.82 / -3.7 / -12.8 at 256 / 1k / 4k / 16k tokens, i.e. systematically negative (the scorer gives the sampled tokens lower probability than the sampler did), and faster than the square-root growth of a random walk; B fp32 stays near zero-mean (mean |S| 0.03 / 0.05 / 0.10 / 0.19) and grows like a random walk; 3 of 8 B-fp32 sequences leave the clip at 16k tokens. Every A and B-bf16 sequence at 4k and 16k is outside the clip (one A-fp32 miss at 4k: 15/16).
3. **Q4.** Plugin sampler is 2.9-4.8x slower than vLLM bf16 in tok/s, at batch 64 and 4k tokens 19.1k vs 4.0k tok/s. As a scorer, the engine runs 6.5k tok/s on the GSM8K rollouts: 19x slower than HF bf16 (122k) and 2.6x slower than HF fp32 (17k); at 4k-token sequences 3.6k vs 94k / 12.8k.

## Deviations and caveats

- **Prediction in the design doc (C = 0; B largest; A in the middle) holds only for C.** B is not the largest: with a bf16 scorer it is slightly below A, with an fp32 scorer it is far below A. Integer arithmetic with 16-bit fixed-point activations sits much closer to fp32 than bf16 does. The doc's expectation that the activation quantization would add error for B did not show up; the residual B-fp32 error 0.0012 is the sum of fp32 weight rounding (<= 6e-8 relative), float vs integer attention/softmax/SiLU/RoPE, and activation quantization, and is not split here.
- A and B run vLLM/HF in **bf16 on weights that are not bf16-exact** (see Setup): the bf16 arms use the int8 weights plus bf16 rounding of those weights. A fp32 vLLM sampler arm was not run.
- Existing rl_mismatch outputs were not found on 141, so A_orig and C were rerun with the same prompts and seeds. The C sampler token count (215,899) and zero mismatch match the earlier report; A_orig tokens differ (216,416 vs 216,083) because `max_model_len` is 2048 in the old run and prompt + 264 here, which changes vLLM's batching and kernels. A_orig numbers (0.0252 / 0.22% / 74.7%) agree with the old table (0.0251 / 0.22% / 74.4%).
- Q1 reward is 1-2% (base model, 2-shot, text up to the first follow-on "Question:" graded); accuracy is not a result of this run.
- HF fp32 at 16384 tokens ran out of memory (sdpa with GQA falls back to the O(S^2) math kernel); those two scorer runs used `--repeat-kv` (K/V repeated, memory-efficient kernel). Shorter fp32 runs use the default path. Both are fp32 attention; no difference was measured between them.
- Engine scoring of sequences beyond 2048 positions needs RoPE rows beyond the constants' 2048: they are rebuilt with the plugin's `_extend_rope` rule (first 2048 rows asserted bit-identical). The first 4k engine run failed on this and was rerun after the fix.
- The standalone engine scoring was checked on a 4-sequence smoke set against the plugin's logits hashes (equal) before the runs; the Q2 engine zero comes from the same code path chunked at 256 positions.
- Speed: single GPU, but host 141 is shared with other jobs on other GPUs (CPU and memory bandwidth contention not controlled). Sampler tok/s counts generated tokens over wall time of `generate` (prefill included, prefix caching on, `ignore_eos`, no hash/log-prob logging in the plugin). Scorer tok/s counts prompt + response tokens through the network; the log-softmax over the response positions is included, row hashing and host transfer are not. The batch-1 plugin 256-token run had one slow first rep (137 tok/s, others 215); median is reported.
- Q2 has 8-64 sequences per length; fractions at 16k are out of 8. Per-token numbers are on 131k to 16k tokens and are stable; sequence-level fractions at long lengths are coarse.
- Scope: one 1B model, base-model continuation of a few-shot prompt, T = 1. Mean |d| for the bf16 arms is dominated by kernel numerics, not by length; nothing here says how a different model, a real long-reasoning distribution, or a trained policy would behave.
