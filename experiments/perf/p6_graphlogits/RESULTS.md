# p6_graphlogits: LM head inside the decode CUDA graph

Base `bc60f84`, branch `perf6/graphlogits` at `e0f6519`.

Change: a decode step replayed from vLLM's full CUDA graph now carries the LM head — final norm, LM-head GEMM, and
(behind a device gate) the one-hot greedy pass — inside the graph. The host sets the gate per replay from the running
requests, so a greedy decode step launches nothing eagerly in `compute_logits` and vLLM's `Sampler.apply` skips only
the float logits copy, not the integer work. A first warm-up call outside the capture compiles the kernels.

## What moved

`src/bitexact/vllm_plugin/model.py`:
- `_glogits(x, N)` appended to the captured decode forward (`_captured_decode`): `pre_norm` + `_gemm` +
  `tri_sample.head(..., "onehot", gate=self.gate)`. Stores `(head, onehot, greedy)` in `self.gl_side[N]` during
  capture; a non-capture call only marks N warm (`self.gl_warm`).
- `_gl_hook()` wraps `ModelCudaGraphManager.run_fullgraph`: before the replay it calls `_gl_replay(N)`, which pushes
  `(strict, noisy)` into `STASH.gmode` and fills `self.gate` with `1` iff the step is greedy-only.
- `compute_logits` takes the graph's outputs when `STASH.gl` is set and the row count fits (`Head(..., rows=R)`
  slices the captured M down to the real R); otherwise it runs the eager path exactly as before.
- `Head.R` (new) is the row count used; `tri_sample.head`'s kernels now launch on `R` rows over the `M`-row GEMM
  output, and `Head.logits()` returns `[:R]`.
- `_head_kernel` / `_comb_kernel` take a `gate_ptr` and `GATED: tl.constexpr`; with `GATED` and `gate == 0` they
  return before doing anything, which is what makes the pass free for noisy / float steps that reuse the same graph.

`src/bitexact/vllm_plugin/sampler.py`: `n = head.R if head is not None else li.shape[0]` (rows that vLLM actually
has, not the padded graph width).

`tri_sample.py` also keeps the eager semantics for `mode == "none"` (`out = h.c`, no one-hot buffer).

## Correctness

- `BITEXACT_GL_CHECK=1` (`_gl_check`) recomputes the norm, the GEMM and the greedy pass from the hidden states
  vLLM hands to `compute_logits` and asserts `torch.equal` on all three, per graph replay. Measured with a counter
  wrapped around `_gl_check` (`glcount.py`, GPU 2, repo tree): b32_t0 **254 checks, gate on in all 254**; b128_t0
  254 checks, gate on in 254; b1_t0 254/254; b32_t1 254 checks, **gate off in all 254**; b128_t1 254/254 gate off.
  So the check did run against both the gated-on and the gated-off replay path, and no assertion fired. Five separate
  runs also completed with `rc=0` at b1/b32/b128 × t0/t1 with the check on (`go_glchk.log`, `GLCHKDONE`). The
  invariance suite below ran with the check off, so the check is evidence for the graph path's bit-identity on its
  own, and the suite is evidence for the end-to-end hashes.
- `pytest -q tests/`: **133 passed, 2 skipped** (`out/SUMMARY_pytest_spec_gpu3.txt`, 90.57 s).
- 1B invariance suite, base and repo, 14 runs each: every run `token_seqs_diff 0 / positions 576 / missing 0 /
  logit_hash_diff 0`, `compare_exit 0` and `compare_long_exit 0`; all 14 repo `.hash` `cmp OK` against base
  (`out/SUMMARY_suite_gpu3.txt`, `out/SUMMARY_suitecmp_gpu3.txt`).
- Token dumps: `md5` of `base_t0.json` == `repo_t0.json`, `base_mixed/lp` == `repo_mixed/lp` — all four
  `f234bdc8…` (mixed, logprobs) and `13ee521a…` (t0).
- Eager-vs-graph spec check (`tests/vllm/spec_check.py`, k=0 vs k=4, 8 cases): all 8 `IDENTICAL` after re-running
  `rep_B32_T0.7_k4`, which the first pass lost to a `ValueError: Free memory on device cuda:0 (38.25/79.11 GiB)`
  during vLLM's memory profiling (`go_specfix.log`: `try 1 rc=0`, `spec rep_B32_T0.7 IDENTICAL`).
- 8B plugin vs IntLlama reference: `compare_exit 0`, 7 runs all `0/0` after re-running `alone_pc1`
  (`go_s8fix.log`, `S8FIXDONE`; the first attempt hit the same memory-profiling failure).
- 8B batch invariance: `BATCH_INV PASS rows 192` for both the eager and the graph path (`out/bi8`).

## Bench, 1B, 128 in / 128 out, 5 interleaved reps, median (tok/s)

| batch | temp | base | repo | ratio |
|---|---|---|---|---|
| 1 | 0 | 552.0 | 556.4 | 1.008 |
| 1 | 1 | 543.5 | 546.1 | 1.005 |
| 32 | 0 | 13,873.1 | 15,927.5 | **1.148** |
| 32 | 1 | 13,016.2 | 14,239.2 | **1.094** |
| 128 | 0 | 30,104.1 | 33,285.2 | **1.106** |
| 128 | 1 | 29,194.9 | 30,666.8 | 1.050 |

B=1 is unchanged (the step is dominated by the GEMM and the sampler, and 1 row is a single graph bucket); the win
starts where the eager host work per step stops being amortised.

## Profile, 1B, ms per step, 3 reps, median

| | base b128_t0 | repo b128_t0 | base b32_t0 | repo b32_t0 | base b1_t0 | repo b1_t0 |
|---|---|---|---|---|---|---|
| tok/s | 27,128.7 | 28,919.9 | 12,979.2 | 14,601.9 | 566.2 | 572.3 |
| step | 3.503 | 3.199 | 2.145 | 1.877 | 1.665 | 1.637 |
| kernel | 2.7449 | 2.7021 | 1.7139 | 1.7134 | 1.5004 | 1.4880 |
| `sampling_logits` | 0.4038 | 0.1077 | 0.2217 | 0.0834 | 0.1627 | 0.0402 |
| `gpu_idle` | 1.4393 | 1.0874 | 1.1431 | 0.7084 | 0.9566 | 0.5533 |
| host sample | 1.0462 | 0.5811 | 1.0436 | 0.5887 | 1.0225 | 0.5851 |

b128_t1 (noisy steps, gate off, same graph): 25,796.5 -> 27,185.6 tok/s, step 3.749 -> 3.495, `sampling_logits`
0.7284 -> 0.5107. The gain is smaller there but still present, because the LM-head GEMM runs inside the graph even
when the greedy pass is gated off.

The whole win is host-side: kernel time is unchanged (2.7449 -> 2.7021 ms) while `sampling_logits`,
`host_sample_ms_per_step` and `gpu_idle` all shrink. B=128 t0 step drops 0.30 ms, of which ~0.30 ms is
`sampling_logits` (0.404 -> 0.108) plus idle.

## One measurement conflict, stated as is

`experiments/perf/p6_graphlogits/stacks.py` (unprofiled `time.perf_counter` split of the same step) reports
`repo_b128_t1` at `pass_s 0.694` / `step_ms_median 7.878` — double base's 0.400 / 3.805 — while the profiler and the
bench both say repo is faster. Its host split is otherwise consistent (`compute_logits` 0.574 vs 0.562,
`sampler_apply` 0.436 vs 0.443), and `repo_b128_t0` matches base (3.546 vs 3.608), so the 2× shows up as extra
`idle_ms_per_step` (2.090 vs 1.006) in that one temp-1 config. **The bench table above is the number to use; the
stacks 7.878 is not reconciled and is not quoted as a result.**

## Files

`src/bitexact/vllm_plugin/{model.py,sampler.py,tri_sample.py}`,
`experiments/perf/p6_graphlogits/{run.sh,stacks.py,hostsplit.py,agg.sh,check_plug.sh}`.