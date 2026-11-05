# p6_wgmma: sm90 wgmma + TMA int8 GEMM (H800, base bc60f84 vs perf6/wgmma 696ee8f)

Change: `src/bitexact/int8/gemm_sm90.py` adds an int8 GEMM kernel built from Hopper wgmma and TMA descriptors, with a
per-(N, K) table of `(max_M, cfg)` rows. `gemm_skinny.int_mm` asks `gemm_sm90.pick(N, K, M)` first and falls back to
its own paths when the table has no config (or the GPU is pre-sm90). Integers are unchanged by construction: the
accumulations are integer adds, which are exact and order-independent while no output overflows int32
(`K * 127 * 128 < 2^31`), so only the speed changes.

## Microbench (`out/gemm/bench_repo.json`, CUDA-graph replay, HBM-cold rotating weights)

70 `(shape, M)` rows; every row has `old_equal` and `new_equal` true, i.e. `torch.equal` to `torch._int_mm` on random
inputs and on the all `-128` / all `127` extremes. Ratio = old ms / new ms where the dispatch actually takes the new
kernel; when `pick` returns `None` the repo runs gemm_skinny's own path, so old and new are the same code and the ratio
is 1.00. The bench `new` column in those rows times `torch._int_mm`, not the fallback; the ratios below use the
effective dispatch.

| shape (N x K) | best | worst |
|---|---|---|
| qkv1 3072x2048 | 1.00x | 1.00x (no table row) |
| o1 2048x2048 | 1.00x | 1.00x (no table row) |
| gu1 16384x2048 | 1.26x @M=384 | **0.71x @M=8192** |
| dn1 2048x8192 | 3.12x @M=384 | 1.00x @M>=1536 |
| lm1 128256x2048 | 1.37x @M=32768 | 0.97x @M=64 |
| qkv8 6144x4096 | 1.18x @M=8192 | 1.00x @M<=512 |
| o8 4096x4096 | 1.23x @M=384 | 1.00x @M<=256 |
| gu8 28672x4096 | 1.26x @M=384 | **0.63x @M=256** |
| dn8 4096x14336 | 1.49x @M=384 | 1.00x @M<=256 |
| lm8 128256x4096 | 2.09x @M=32768 | 1.00x @M<=128 |

Decode-sized rows (M = 128 / 256 for 1B, M = 64..256 for 8B) mostly stay on the old path, which is the point of the
table. The two real losses are `gu1` at M = 8192 (0.87 -> 1.22 ms) and `gu8` at M = 256 (0.099 -> 0.157 ms); both are
configs the sweep chose, not regressions of the fallback. Prefill-sized rows gain where the table has an entry.

## Decode op drop-out in the bucket classifier

`experiments/perf/p5_prefill/buckets.py::bucket` classifies a GEMM by name: `_INT` matches
`s8|i8|int8|imma|igemm|_int_mm`, so the new kernel (`_wgmma`) lands in `gemm_bf16`. Consequences in every profile
table below: `gemm_int8` counts 0 for the repo tree even though the GEMMs are still int8, and `gemm_bf16` jumps from 0
to the GEMM time. Compare `gemm_int8 + gemm_bf16` between trees, not either column alone.

## Decode profile, 1B B=128 (ms per step, median of 3)

| | base | repo |
|---|---|---|
| kernel | 2.7407 | 2.6919 |
| `gemm_int8` + `gemm_bf16` | 0.000 + 0.000 | 0.000 + 0.000 |
| `sampling_logits` | 0.404 | 0.368 |
| `gpu_idle` | 1.439 | 1.105 |

At B=128 decode the LLM-shape GEMMs are dominated by `_k` (the pre-existing Triton path) and the wgmma table mostly
does not fire, so decode barely moves — consistent with the e2e table below.

## Prefill profile, 1B B=128 x 128 (ms per step)

| | base | repo | ratio |
|---|---|---|---|
| kernel | 133.58 | 125.10 | 1.068 |
| `gemm_int8` (base) / `gemm_bf16` (repo) | 0.00 | 36.34 | — |
| `other` | 122.45 | 77.66 | — |

Prefill is where the lane pays off: the same kernels move from `other` into the new `gemm_bf16` bucket and prefill
kernel time drops 6.4-6.8%. Prefill throughput (prompt tok/s, median of 3, `out/prefill`):

| B x L | base | repo | ratio |
|---|---|---|---|
| 64 x 256 | 105,759 | 112,320 | 1.062 |
| 64 x 512 | 100,052 | 106,053 | 1.060 |
| 64 x 1024 | 83,129 | 90,165 | 1.085 |

## 8B decode profile, B=128 (ms per step, median of 3)

| | base | repo |
|---|---|---|
| kernel | 11.028 | 10.391 |
| `gemm_int8` / `gemm_bf16` | 2.937 / 0.000 | 0.000 / 2.444 |
| `other` | 4.165 | 4.105 |
| `attention` | 2.670 | 2.628 |

8B decode gains 5.8% of kernel time (11.03 -> 10.39 ms) because the table does fire at B=128 for the 8B shapes.

## End-to-end, 1B (128 in / 128 out, T=0, median of 5 interleaved, `out/e2e`)

| batch | base | repo |
|---|---|---|
| 1 | 554.1 | 542.1 |
| 32 | 14,015.8 | 12,410.8 |
| 128 | 30,988.5 | 30,593.9 |

**B=32 looked like an 11% regression, and it is a measurement artefact of one specific config.** The profiler disables
prefix caching (`enable_prefix_caching=False`), the bench does not. Re-running the bench with the flag exposed
(`out/pc.log`, 3 reps each, GPU 3):

| config | base | repo |
|---|---|---|
| `enable_prefix_caching=1` | 14,093.6 / 14,243.8 / 13,763.8 (median 14,093.6) | 13,033.4 / 12,794.1 / 13,111.0 (median 13,033.4) |
| `enable_prefix_caching=0` | 13,511.8 / 13,668.7 / 13,471.2 (median 13,511.8) | 13,520.4 / 7,271.5 / 13,683.0 (median 13,520.4) |

With prefix caching off the two trees are equal. With it on, the repo tree is 7.5% slower — and the profiler with
prefix caching forced on (`out/profpc1_b32`, `--pc 1`) shows where: **the prefill step, not decode**.

| B=32, pc=1 | base | repo |
|---|---|---|
| prefill step ms | 12.06 | 37.94 |
| decode step ms (median) | 2.202 | 2.181 |
| decode kernel ms | 1.675 | 1.685 |

The extra ~26 ms is one-time: running the same batch three times in one process (`bench_pc3.py`, two reps each)

| | pass 0 | pass 1 | pass 2 |
|---|---|---|---|
| repo r1 | 12,625.0 | 14,033.9 | 14,058.1 |
| repo r2 | 13,035.8 | 14,339.1 | 14,389.0 |
| base r2 | 14,285.3 | 14,404.0 | 14,423.0 |

Repo steady state is within 1.4% of base; pass 0 pays ~27 ms once (first-touch of the TMA descriptors / wgmma kernel
inside the prefix-caching codepath, which the bench's timing pass is the first to hit). Disabling the new kernel
(`gemm_sm90.sm90 = lambda: False` in the repo tree) removes the gap: 13,487.2 / 14,237.5 vs 12,686.1 for repo with it
on. So the cost is the wgmma kernel's first call, not a steady-state regression.

The `enable_prefix_caching=0` repo set has one bad point (7,271.5) — the GPU was clean per `quiet_check` but the run
was 4.3 ms/step instead of 3.0; median is unaffected.

## End-to-end, 8B (128 in / 128 out, T=0, median of 5 interleaved, `out/e2e8b`, `out/e2e8c`)

| batch | base | repo | ratio |
|---|---|---|---|
| 1 | 180.3 (n=5) | 179.9 (n=5) | 0.998 |
| 32 | 4,959.8 (n=5) | 4,989.0 (n=5) | 1.006 |
| 128 | 10,303.5 (n=5) | 10,776.5 (n=5) | **1.046** |

B=128 8B is the one place the lane gains end to end: +4.6%, from the 8B decode profile's 11.03 -> 10.39 ms of kernel
time. B=1 and B=32 are flat.

The earlier `out/e2e8` numbers were contaminated: a `pass3_8b.sh` process was sharing GPU 1 with the e2e runs, which is
why one repo set read 8,047 and another 10,851 with `sec` swinging between 1.51 and 2.04 for the same work. Re-running
both trees alone on an otherwise idle GPU (`go_e2e8b.log`, `go_e2e8c.log`, 5 reps each) gives the tight sets above —
repo B=128 10,745 / 10,747 / 10,781 / 10,776 / 10,780 against base 10,308 / 10,304 / 10,270 / 10,303 / 10,301. The
`out/e2e8` figures should not be quoted.

The same isolation check with prefix caching explicit (`bench_pc.py`, 3 reps, GPU 3) confirms the direction and shows
the cache matters:

| 8B B=128 | base | repo |
|---|---|---|
| `enable_prefix_caching=0` | 7,330.4 | 7,811.1 (1.066) |
| `enable_prefix_caching=1` | 10,279.4 | 10,751.8 (1.046) |

## Regression, stated plainly

- `gu1` M=8192: 0.87 -> 1.22 ms (0.71x). Real, in the table, not fixed here.
- `gu8` M=256: 0.099 -> 0.157 ms (0.63x). Real, in the table.
- `lm1` M=64: 0.221 -> 0.229 ms (0.97x). Marginal.
- B=32 1B with prefix caching on: one-time +27 ms on the first batch of a process; steady state −1.4%.

## Bit identity

- `pytest -q tests/` on the repo tree: **135 passed, 2 skipped** (`out/pytest/pytest_repo.log`, 11:53, GPU 0). An
  earlier 10:42 run had 7 failures from a `NameError` in a half-synced tree; the 11:53 run is the one on `696ee8f`.
- 1B invariance suite, base and repo, each 14 runs: every run `token_seqs_diff 0 … logit_hash_diff 0`, `compare_exit 0`
  and `compare_long_exit 0`; all 14 repo `.hash` files `cmp` equal to base's (`out/SUMMARY_e2e_suite_gpu3.txt`, 13:28-13:44).
- Token dumps (`mixed` / `t0` / `lp`): all three `IDENTICAL` between trees.
- 8B plugin vs IntLlama reference (`out/s8`): `compare_exit 0`, all runs `0/0`.
- 8B batch invariance (`out/bi8`): `BATCH_INV PASS rows 192` for both graph and eager.

## Files

`src/bitexact/int8/gemm_sm90.py` (new kernel, `TABLE`, `pick`, `sm90`, `gemm_dispatch`),
`src/bitexact/int8/gemm_skinny.py` (`int_mm` asks `gemm_sm90.pick` first), `tests/test_gemm_sm90.py`,
`experiments/perf/p6_wgmma/{run.sh,bench_gemm.py,gt.py,sweep4.py,diag.py,mk_table.py}`.