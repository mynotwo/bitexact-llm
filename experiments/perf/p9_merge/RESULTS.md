# p9_merge: combined verification of the four performance lanes

Four lanes merged into `main` after `175ff6c`: `perf6/seeds` (`4f3ec84`), `perf6/graphlogits` (`c904a49`),
`perf6/wgmma` (`6387c33`), `perf8/exactfuse` (`bcb2fbf`). Each lane verifies itself; this lane re-runs the whole
gate on the merged tree, because a lane that is bit-identical alone can still change a neighbour's kernels,
dispatch tables or graph capture.

Trees: `base` = `175ff6c` (the main commit the lanes forked from), `repo` = `bcb2fbf` (main with all four merged).
Both trees are exports of the same repository, run on the same GPUs in the same sessions (`run.sh`,
`readme_bench.sh`; summaries under `$S/out/SUMMARY_*` on 141, `S=/data41/audreymao/perf9_20261105/merge`).

## Correctness

### pytest, both trees, same environment

Environment carries every constant and model the tests can use, so a test that skips is a real skip, not a missing
path: `BITEXACT_INT8_CONST`, `BITEXACT_INT8_CONST_8B`, `BITEXACT_EXACT_CONST`, `BITEXACT_MODEL` (HF Llama dir for
the 2-layer reference build) and `BITEXACT_VLLM_MODEL` (plugin model dir for the overflow check).

| Tree | Result |
|---|---|
| `base` `175ff6c` | **138 passed, 1 skipped** in 164.40 s |
| `repo` `bcb2fbf` | **147 passed, 8 skipped** in 163.13 s |

Every skip on both trees has the same reason, `exact constants only`: the int8 parameter instances of
`tests/test_vllm_exact_gemm.py` (one on `base`) and `tests/test_vllm_exact_fuse.py` (seven on `repo`). Those
instances are spent because the kernels they compare against are exact-mode only. Collection: 139 tests on `base`,
155 on `repo`; the 16 new ones in `tests/test_vllm_exact_fuse.py` are 9 that pass and 7 that skip, so the repo
tree runs every test the base tree runs and passes all of them — nothing regressed.

`BITEXACT_VLLM_MODEL` matters: without it the overflow check is skipped on both trees, which hides exactly the
difference above.

### Invariance suite, 14 runs per tree

`tests/vllm/run_tests.sh` on each tree, 10 short-target runs (alone, busy mixed batches with 40 fillers, chunked
prefill budgets 64 / 256 / 512, prefix cache on and off, 160-block KV cache with 5 preemptions, full eager with
`max_num_seqs` 128, vLLM graphs) and 4 long-target runs:

- `base`: 14 runs, every one `token_seqs_diff 0`, `logit_hash_diff 0`, `missing 0`, `inconsistent_repeats 0`,
  `compare_exit 0`, `compare_long_exit 0`.
- `repo`: identical, 14 runs, all `0 / 0 / 0 / 0`, both compare exits 0.
- 14 `.hash` files compared `base` vs `repo`, one per run: **14 `cmp OK`**.

### Token dumps from the production path

`experiments/perf/p4_merge/tokens.py` on both trees, three modes — mixed temperatures, T = 0 only, and with
logprobs — compared with `cmp` on the JSON:

| Dump | Result |
|---|---|
| mixed temperatures | IDENTICAL |
| T = 0 | IDENTICAL |
| logprobs | IDENTICAL |

The T = 0 dump is the one that used to differ: `sampler.py::IntSampler.add_request` returns
`bool(self.strict[req_idx] or self.tn[req_idx] != 0)`, so a pure greedy non-strict request reports `False`, vLLM's
`Sampler.apply_sampling_params` skips the whole batch, `IntSampler.apply()` never runs, and the old `model.py`
handed `torch.empty` to `gumbel_sample`, whose argmax read uninitialised memory. The bug is in `base` too; the
exactfuse lane fixes it by making `model.py` always return real logits (`tri_logits.argmax_float` `mode="onehot"`).
Both trees carry the fix here, which is why the dump matches and why the suite above is a test of the lanes and not
of the bug.

### abc_rl plugin check

The fourth lane is documentation only (`experiments/abc_rl/RESULTS.md`: C_plugin at 4 seeds, 0.150 ± 0.017 at
step 200 against 0.177 ± 0.024 for the standalone integer arm, Welch p 0.13), so the gate for it is the plugin
check its numbers rest on: `experiments/abc_rl/check_plug.py` on `repo`, four updates — `rc=0`, and in every
update `seq_tokens_differ`, `sampler_lp_differ` and `rescore_lp_differ` are 0 (`n_tok` 14,470 / 13,571 / 12,283 /
13,734).

## Bench

`readme_bench.sh` runs the 24 cells the README tables need: exact / int8 / bf16 stock / bf16 batch-invariant, at
batch 1 / 32 / 128, on 1B and 8B. One process per cell (`tests/vllm/bench.py`, 128 in / 128 out, T = 0, one timed
`generate` after a warm-up pass), five repetitions, cycling configurations so drift hits every column. 1B on GPU 2,
8B on GPU 3, never two jobs on one GPU; before each cell `wait_gpu` blocks until the card reads 0 MiB and has no
compute process. Summary: `$S/out/rd/*.log`, aggregation `readme_bench.sh agg`.

1B, output tok/s, n = 5, median [min - max]:

| Configuration | Batch 1 | Batch 32 | Batch 128 |
|---|---|---|---|
| integer plugin, exact mode | 254.2 [237.1 - 254.6] | 6,800.7 [6,748.6 - 6,903.9] | 13,939.7 [13,904.6 - 13,995.0] |
| integer plugin, int8 mode | 548.3 [543.4 - 579.5] | 14,639.9 [14,321.9 - 14,693.1] | 33,462.6 [33,159.8 - 33,833.2] |
| vLLM bf16, stock | 612.3 [605.5 - 614.2] | 16,128.2 [16,085.7 - 16,212.1] | 33,830.9 [33,472.6 - 34,392.6] |
| vLLM bf16, batch-invariant + `TRITON_ATTN` | 213.8 [212.4 - 214.5] | 7,549.1 [7,476.4 - 7,572.0] | 34,487.4 [34,009.8 - 34,606.9] |

8B, same protocol:

| Configuration | Batch 1 | Batch 32 | Batch 128 |
|---|---|---|---|
| integer plugin, exact mode | 65.5 [65.4 - 65.5] | 1,362.9 [1,356.2 - 1,370.9] | 3,087.8 [3,084.9 - 3,095.9] |
| integer plugin, int8 mode | 180.0 [168.2 - 180.2] | 4,978.4 [4,974.4 - 5,098.9] | 10,781.4 [10,755.2 - 10,789.3] |
| vLLM bf16, stock | 151.5 [151.3 - 154.0] | 4,532.5 [4,508.5 - 4,625.6] | 14,063.9 [14,000.7 - 14,280.7] |
| vLLM bf16, batch-invariant + `TRITON_ATTN` | 49.8 [49.8 - 50.2] | 1,744.6 [1,741.1 - 1,747.4] | 10,675.1 [10,643.0 - 10,752.3] |

Spread: at most 2.7% at batch 32 and 128 across all 24 cells. At batch 1 it is 6.9% (1B exact) and 6.7% (8B int8,
one slow point at 168.2 against 180.0-180.2 for the other four), while 1B int8 reads 6.6% only because of a single
579.5 run; its other four sit at 543.4-550.7. A step at batch 1 is about 4 ms, so that is measurement noise, not a
drifting machine: the median of reps 1-2 against reps 3-5 moves by at most 1.6% in any cell (1B int8 batch 32,
14,692 -> 14,464), and by under 1% everywhere else.

The 8B int8 batch-128 cell agrees with the lane bench that produced it (`p6_wgmma` measured 10,776.5 against
10,781.4 here). The exact batch-128 cell is 3,087.8 against 1,124 in
`p7_exactplugin` RESULTS: 2.7x, from the fused elementwise kernels of `p8_exactfuse` (elementwise 3.55 -> 1.04 ms
per decode step, 633 -> 218 kernels) and the 8B rows of the `p6_wgmma` GEMM table.

### Exact-mode profile after the merge

`p8_exactfuse/prof_vllm.py` on both trees, 1B exact, batch 128, prefix caching off, profiler window on decode steps
8-23 and on the first prefill step, back to back on GPU 2 (`$S/out/prof/ab_base.json.json` and `ab_repo.json.json`;
the script appends `.json` to its `--out` name). Kernel time per step, ms:

| | base `175ff6c` | repo `bcb2fbf` |
|---|---|---|
| prefill kernels | 642 | 227 |
| prefill GPU kernel time, total | 956.69 | 667.16 |
| prefill `elementwise` | 374.81 | 78.70 |
| prefill host step (enqueue of the 128 x 128 prefill) | 28.17 | 12.10 |
| decode step | 10.89 | 8.39 |
| decode `elementwise` | 3.55 | 1.04 |
| decode `gemm_int8` | 4.93 | 4.91 |
| decode `sampling_logits` | 1.22 | 1.23 |
| decode `attention` | 0.75 | 0.74 |
| decode kernels | 633 | 218 |

The repo decode step's 8.13 ms of kernel time is those buckets plus 0.20 of `other` and 0.01 of `memcpy`; on `base`
the same four buckets plus 0.05 `other` and 0.01 `memcpy` make 10.50. Only `elementwise` moves (2.51 ms), which is
what `p8_exactfuse` replaces; `gemm_int8` (4.93 -> 4.91), `sampling_logits` (1.22 -> 1.23) and `attention`
(0.75 -> 0.74) are unchanged within noise.

The prefill rows say the same thing at the other end. That step runs 642 kernels on `base` and 227 on `repo`; its
elementwise share drops 374.81 -> 78.70 ms while the GEMM share (542.46 -> 539.24) does not move, so GPU kernel time
falls 956.69 -> 667.16 ms and the host enqueue of the step 28.17 -> 12.10 ms. `other` rises 1.34 -> 11.37 ms only
because `_attn_out_kernel` and `_qkv_kernel` move into that bucket. Read the decode step and its ratio (1.30x,
matching the lane's own A/B of 10.927 -> 8.332), not the profiler's `tok_per_s` line: it reads 9,351 here where
`bench.py` reads 13,913 for the same tree and batch, because the profiler turns prefix caching off and `bench.py`
does not.

`pcbench.sh` measures that tax directly (`bench_pc.py` is `tests/vllm/bench.py` with
`enable_prefix_caching=(PC == "1")`, two interleaved reps per configuration, logs in `$S/pc/`; medians):

| cell, batch 128, T = 0 | pc = 1 | pc = 0 | lost |
|---|---|---|---|
| 1B exact | 13,860 | 9,359 | 32.5% |
| 8B exact | 3,097 | 1,997 | 35.5% |
| 8B int8 | 10,783 | 7,816 | 27.5% |
| 1B int8 | 33,330 | 30,198 | 9.4% |
| 1B bf16 | 33,595 | 33,611 | none |

A second run of the same script reproduces every row within 0.5 points (32.6% / 35.5% / 27.2% / 9.8%, and -2.7% for
1B bf16, i.e. still noise; logs in `$S/pc2/`, driver `$S/out/pc2_go.log`, 21:01-21:07), so the table is a property
of the configuration, not of one session.

The `pc=0` 1B exact point sits within 0.1% of the profiler's 9,351, and the 8B profile's own 7,787 tok/s sits
within 0.4% of the 8B int8 `pc=0` point, so each configuration is self-consistent. The tax grows with prefill cost,
and it is why profile numbers and bench numbers are comparable only inside one configuration. The process mode is
not a factor either: `readme_bench.sh` sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` for every bench run too, and a bench
run with it at the default `1` reads 13,758 against 13,851 on that mode (0.7%). Both trees here are profiled the
same way, so the ratios stand.

The same profiler on 8B int8 batch 128 (`$S/out/prof/{base,repo}8b_b128.json`, GPU 2), decode steps 8-23, kernel
ms per step:

| | base `175ff6c` | repo `bcb2fbf` |
|---|---|---|
| decode step | 11.75 | 11.27 |
| kernel | 11.23 | 10.46 |
| GEMM, all calls | 3.32 (33 per step, cutlass s8) | 2.74 (33 per step, `_wgmma`) |
| attention | 2.72 | 2.65 |
| elementwise | 0.78 | 0.77 |
| `sampling_logits` bucket | 0.54 | 0.11 |
| kernels per step | 388 | 388 |

Kernel names are identical between the two traces except the GEMM: 512 cutlass s8 calls plus 16 cutlass 128x256
calls (the LM-head GEMM) on `base`, 528 `_wgmma` calls on `repo`. Read the two middle rows together, because the
LM-head GEMM changes bucket: on `base` it sits in `sampling_logits` (0.34 of that 0.54), and on `repo` it is one of
the 33 `gemm_bf16` calls, so the GEMM row is the full 3.32 -> 2.74 (0.58 ms, 18%) and the bucket row is what is
left in sampling: 0.54 -> 0.11. The single prefill step of the same batch moves with it: 746 -> 669 ms of kernel
time, 402 kernels on both trees, because prefill runs the same GEMM path (497 -> 421).

The gain is the GEMM (`p6_wgmma`) plus the head moving into the decode graph (`p6_graphlogits`); attention and
elementwise barely move because 8B int8 has no exact-mode work in it. Step and end-to-end numbers here are the
profile-only configuration, so they are lower than the README's 10,781 tok/s for the same tree.

The same profiler on GPU 1 reads 10.5 ms decode and 7,340 tok/s where GPU 2 reads 8.4 ms and 9,351 on the same
binary, and `bench.py` reads 11.2-12.1k on GPU 1 against 13.9k on GPU 2/3. GPU 1 is the outlier (it also hosted the
earlier phase runs); all README numbers come from GPU 2 and 3.