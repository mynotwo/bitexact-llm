# perf3/spec: ngram speculative decoding on the integer plugin

Conclusion: bit-identity PASS (9/9 workload pairs token-identical to non-spec), B=1 speedup 1.5-2.0x on copy-heavy prompts and 1.4-1.5x on the random bench workload, B=32 slower (0.4-0.9x). Not worth merging as a default; fine as opt-in for low-batch latency.

## What was needed (vLLM 0.32, V2 model runner, built-in `ngram` proposer, k=4, prompt_lookup 2..5)
1. Sampler (`sampler.py`): with drafts, a request owns 1+k logits rows. vLLM calls our `LogitsProcessor`, which overwrites every row with the integer-sampled token (one-hot). Row j is now keyed by its own position (seed, t = pos+1-prompt_len), taken from the device `ctx.pos` (one host sync per spec step, spec path only). vLLM's rejection sampler then sees a one-hot target and accepts a draft iff it equals our token at that position; the bonus/resampled token is also our token. So the accept rule is exactly "draft == what plain decoding samples there" for T=0 and T>0, and no vLLM rejection code needed overriding.
2. Model (`model.py`): multi-token verify batches (q = 1+k rows per request) now take the graph path: block table is expanded per row (row i -> request i // q); the existing integer paged attention takes per-row positions (causal per row), so multi-query decode needed no kernel change. Prefill-shaped uniform q batches (vLLM sees them as `plan.pre`) are routed to the same path. Also: a dummy `torch.compile` at model init so Inductor's one-time pattern init (a host-to-device copy) does not land inside vLLM's CUDA graph capture (this crashed capture with spec on).
3. Async scheduling is disabled by vLLM for ngram (host proposes drafts), which is part of the cost.

## Bit-identity (GPU 7, vLLM graphs on, k=0 vs k=4, same seeds, token lists compared)
suite (12 targets, T 0/0.7/1.0 mixed, 96 new): identical. rep (copy-heavy) and bench (random prompts), B=1 and B=32, T=0 and T=0.7: identical. 9 of 9 pairs. (An eager run of suite and rep B=8 at T=0/0.7 was identical too.) Logit hashes were not compared (tokens only): not verified.
Not run: the 14-job invariance suite (run_tests.sh), preemption/chunked-prefill with drafts, long context.

## tok/s (one quiet rep, k=0 vs k=4; second rep partly finished, k=0 within 10% of first; host noise 15-20%)
| workload | T | B | k=0 | k=4 | accepted/drafted |
|---|---|---|---|---|---|
| rep (copy) | 0 | 1 | 480 | 729 | 47% |
| rep | 0.7 | 1 | 399 | 818 | 76% |
| bench | 0 | 1 | 470 | 671 | 44% |
| bench | 0.7 | 1 | 455 | 635 | 56% |
| rep | 0 | 32 | 10015 | 9492 | 58% |
| rep | 0.7 | 32 | 6185 | 4926 | 42% |
| bench | 0 | 32 | 11111 | 6886 | 45% |
| bench | 0.7 | 32 | 9653 | 4195 | 16% |
| suite (12 mixed) | mixed | 12 | 3914 | 2087 | 18% |
(B=1 k=0 is ~470 here vs 388 in the brief, the brief's number is the B=1 128-in bench from another run.) The random "bench" prompts accept ~45% because the int8 model's own output loops, so ngram hits on generated text.

## Why B=32 loses
A verify step runs 5x the rows through the full model and LM head (GEMM + sampling are per row), and every step needs host proposals and a device sync for t, with async scheduling off. At B=1 the step is launch-bound so extra rows are nearly free; at B=32 the rows are real work and accepted tokens (~1.5-2.5 per step at 45-58% acceptance) do not repay 5x.

## Not done / not verified
Draft model / EAGLE proposers; adaptive k by batch size (would turn B=32 off, giving B=1 gains only); profiler run at B=128; 3-rep medians (1 full rep plus partial second); logit-hash comparison.

Code: branch perf3/spec; tests/vllm/spec_check.py (harness), experiments/perf/p3_spec/go.sh.
