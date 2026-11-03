# p2_sample: fused LM-head post-processing, greedy argmax and integer Gumbel sampling

Branch `perf2/sample` on 971a3e1, H800 GPU 3, 1B int8 model, vLLM 0.32 plugin.

## Conclusion

- Bit identity: PASS. 14/14 suite hash files identical to base (`run_tests.sh`, logit hashes + token sequences, incl. preempt, eager, vLLM graph, long-context), `pytest` 27 passed / 1 skipped, and the new production-path token dumps (128 requests, temperatures 0/1/0.7/0.5/1.5/2, T=0 only, and logprobs=1) are identical to base.
- GPU time at B=128 (profiler, ms/step, `sampling_logits` bucket without the 0.207 LM-head GEMM): T=0 0.394 -> 0.215 (-0.18 ms, about 3.4% of the 5.3 ms wall step); T=1 1.159 -> 0.603 (-0.56 ms). Kernel sum per step: T=0 4.606 -> 4.425, T=1 5.362 -> 4.812.
- tok/s: T=0 B=128 +7.7% (median of 3, but runs are noisy: base 20.1-23.2k, branch 17.9-24.5k); T=1 B=128 +0.2%: the T=1 step is host bound (wall 9.2 ms vs 5.4 ms kernel), so the saved 0.56 ms of GPU time does not show up in tok/s. B=1 T=0 reads 0.92x, which is inside run-to-run spread (base 324-382, branch 317-376); not verified further.
- Verdict: the T=0 saving reaches the 3% threshold on GPU time (3.4%) but the tok/s gain is not separable from noise; T=1 gain is real on the GPU and invisible in tok/s. Left unmerged for the coordinator to decide.

## What changed

New `src/bitexact/vllm_plugin/tri_sample.py` plus wiring in `model.compute_logits` and `sampler.IntSampler`. The int64 logits [B, 128256] are never stored:

- `_head_kernel`: reads the LM-head GEMM output (int32 limbs), does segs.post (join limbs, `rsh(acc*m, fmsh)`, `rsh(t*mant, 24-FR-e)`) inline, produces the row first argmax in one pass (per-chunk partial max/index, `_comb_kernel` combines). Output to vLLM is one-hot float logits (-inf, 0 at argmax) when no running request needs real float logits; otherwise the exact float copy `li * 2^-16` as before.
- `_noise_kernel` (T>0): same pass plus PCG64 word i (closed form from gpu_pcg), integer Gumbel noise, `rdiv(l*td, tn) + g`, first argmax. Rows with tn=0 take plain logits, so mixed batches are one kernel. Seed states, tn, td go to the device as one pinned int64 [M, 6] tensor. When any running request has T>0 compute_logits skips its own pass and returns an uninitialised buffer; the sampler rewrites every row.
- Requests that read or edit vLLM's float logits (logprobs, prompt_logprobs, penalties, logit_bias, allowed/bad words, min_tokens, structured output, extra_args) are detected per slot (`_strict`), only for running requests (`_request_state`, found once through gc since vLLM gives no finish callback), and keep the old float path. `BITEXACT_HASH_LOG` also keeps it.
- Kernels removed from the step: Inductor segs.post for the LM head (0.09), `_part_kernel` (0.07), memcpy/fill for the float copy and `fill_(-inf)` (partly), PCG kernel (0.06), the two Inductor reduction kernels for argmax of noisy logits (0.26), Inductor sampling poi kernel (0.45 at T=1).

Same integers: every element runs the same int64 operations in the same order; ln_q's bitlen is 64 - clz (exact); rdiv is the floor of (2a+b)/(2b) with an explicit remainder correction; argmax is the exact first maximum. Unit tests (`tests/test_vllm_kernels.py::test_tri_sample_*`) compare against eager segs.post / `sampling.sample` / `sampling.gumbel` for M=1/5/7/128, large and small logit scales, ties, and mixed temperatures including (tn, td) = (100, 1), (1, 100).

## Profile (B=128, 128 in / 128 out, ms/step, `sampling_logits` bucket)

| | T=0 base | T=0 branch | T=1 base | T=1 branch |
|---|---|---|---|---|
| LM-head GEMM | 0.207 | 0.207 | 0.207 | 0.207 |
| everything after the GEMM | 0.394 | 0.215 | 1.159 | 0.603 |
| bucket total | 0.601 | 0.422 | 1.365 | 0.810 |
| kernel sum per step | 4.606 | 4.425 | 5.362 | 4.812 |

Remaining post-GEMM time, T=0: `_gumbel_sample_kernel` 0.089 (vLLM's own sampler on the one-hot logits; not touched), `_head_kernel` 0.074. T=1: `_noise_kernel` 0.34 (PCG + two ln_q table lookups per logit, compute bound), vLLM `_gumbel_sample_kernel` 0.13, DtoD memcpy 0.05.

## tok/s (3 interleaved base/branch runs each, median; bench.py 128 in / 128 out)

| B | T | base | branch | ratio | runs base / branch |
|---|---|---|---|---|---|
| 1 | 0 | 375 | 344 | 0.92 | 324,382,375 / 317,376,345 |
| 32 | 0 | 9,447 | 9,621 | 1.02 | 8038,9613,9447 / 8027,9621,9757 |
| 128 | 0 | 21,052 | 22,670 | 1.08 | 20098,21052,23177 / 17865,22670,24523 |
| 1 | 1 | 370 | 383 | 1.04 | 373,342,370 / 384,383,372 |
| 32 | 1 | 8,880 | 9,335 | 1.05 | 8948,8880,8399 / 9132,9338,9335 |
| 128 | 1 | 18,288 | 18,332 | 1.00 | 16400,18288,18841 / 17666,18801,18332 |

The H800 host is shared with other jobs on other GPUs; spread between runs of the same tree is up to 15%.

## What did not work / not done

- A sticky strict flag (set once by any request with logprobs) kept the float path on forever; vLLM's warm-up requests set every feature on, so it was always on. Replaced by per-slot flags restricted to running requests.
- vLLM's remaining greedy `_gumbel_sample_kernel` on the one-hot logits (0.09-0.13 ms) is still launched; skipping it would need changes inside vLLM's sampler (out of scope).
- The T=1 step is host bound (per-row `np.random.PCG64(SeedSequence(...))` seeding for 128 rows in `row_state`, plus Python bookkeeping); not optimised here. Likely the next lever for T>0 tok/s.

## Not verified

- tok/s gain is within noise at B=128 T=0; B=1 T=0 0.92x not re-measured beyond 3 runs.
- Hash logging (`BITEXACT_HASH_LOG`) recomputes logits eagerly via `Head.logits()`, so the 14 suite hashes validate the sampler/argmax tokens and the eager reference, while the fused logits path is validated by the unit tests and the token dumps, not by the logit hashes.
- Speculative decoding unsupported (as before). Other GPUs / models not run.

## Reproduce

`bash experiments/perf/p2_sample/run.sh sync`, then `STAGES="tokens pytest bench prof suite" GPU=3 bash experiments/perf/p2_sample/run.sh start` (`TEMP=1` for the T=1 profile; `tokens.py` is the base-vs-branch token dump).
