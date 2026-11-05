# p8_exactfuse: fused element-wise kernels in exact mode

Base `daac192` minus the fix below, branch `perf8/exactfuse`.

Change: the ~10 Inductor kernels that exact mode runs between two linears (embedding add + final norm + quant, RoPE +
per-head quant + paged KV store, attention divide + quant, silu + quant) become four Triton kernels in
`src/bitexact/vllm_plugin/exact_fuse.py`. Every element goes through the same int64 ops as `exact/ops.py` — the same
shifts, floor-division correction, `bitlen`, exact integer max/sum — so the quantized integers and bound flags are
the eager stages' bit for bit. `exact_run.py` routes the exact path through them.

## Correctness

- `pytest -q tests/`: **131 passed, 22 skipped** (`out/SUMMARY_pytest_gpu0.txt`, repo tree, 134.27 s). Base tree:
  124 passed, 15 skipped — the 7-run difference is `tests/test_vllm_exact_fuse.py`, which compares every fused kernel
  against the eager `exact/segs.py` stage.
- 1B invariance suite, base and repo, 14 runs each: every run `token_seqs_diff 0 / logit_hash_diff 0`,
  `compare_exit 0`, `compare_long_exit 0`; all 28 `.hash` `cmp OK` (`out/SUMMARY_suite_suitecmp_gpu1.txt`).
- Token dumps: t0 `3c2c4c29…` identical base/repo, mixed and logprobs `901e2814…` identical base/repo.
- 8B exact vs the IntLlama reference: `compare_exit 0`.

### A pre-existing bug the exact path did trip over

The first spec/t0 dump differed between base and repo, and the cause is in base, not in the fusion. In
`sampler.py::IntSampler.add_request` the return value is

```python
return bool(self.strict[req_idx] or self.tn[req_idx] != 0)
```

so a pure greedy, non-strict request reports `False` → vLLM's `Sampler.apply_sampling_params` sees
`np.any(self.needs_logits_processing[idx_mapping_np]) == False` for the whole batch and returns the logits unchanged
→ `IntSampler.apply()` never runs → the old `model.py` path returned `torch.empty(...)` → vLLM's `gumbel_sample`
argmaxes uninitialised memory. Base (`175ff6c`) has the same bug; it is timing-dependent, which is why t0 tokens were
only sometimes garbage.

Fix: `tri_logits.argmax_float` gained `mode` in `{"none", "float", "onehot"}` — `"onehot"` stores `0.0` at the
selected index and `-inf` elsewhere (one finite value per row), `"float"` stores `x * 2**-16`, `"none"` stores the
exact int `c` untouched. `model.py` now always returns real logits:

```python
if not STASH.strict and STASH.sampler.any_noisy():
    _, STASH.greedy = tri_logits.argmax_float(li, mode="none")
    return torch.empty(li.shape, dtype=torch.float32, device=li.device)
f, STASH.greedy = tri_logits.argmax_float(li, scale=2.0 ** -mm.FR,
                                           mode="float" if STASH.strict else "onehot")
return f
```

`torch.empty` survives in the noisy branch because vLLM overwrites those rows with the sampled float logits; it is
still the value the sampler needs, and that branch was correct in base.

To separate the fusion from this bug, **both trees carry the same patch**, and the spec and t0 dumps then agree
(`out/tok2/base_t0.json` = `out/tok2/repo_t0.json`, `3c2c4c29…`, `tokens t0 IDENTICAL`).

## Bench, 1B, 128 in / 128 out, 5 interleaved reps, median tok/s

Two e2e blocks, same GPU, same tree: block 0 before the one-hot patch, block 1 after (`SUMMARY_e2e_gpu2.txt`).

| cfg | base-equivalent (pre-fix) | post-fix | delta |
|---|---|---|---|
| exact b1 t0 | 255.3 | 254.2 | −0.4% |
| exact b1 t1 | 250.2 | 249.7 | −0.2% |
| exact b32 t0 | 6,794.5 | 6,802.6 | +0.1% |
| exact b32 t1 | 6,370.0 | 6,371.2 | +0.0% |
| exact b128 t0 | 13,907.9 | 13,906.1 | −0.0% |
| exact b128 t1 | 12,463.2 | 12,472.8 | +0.1% |
| int8 b1 | 556.3 | 553.6 | −0.5% |
| int8 b32 | 13,976.4 | 13,994.9 | +0.1% |
| int8 b128 | 30,622.6 | 30,765.1 | +0.5% |

`bf16` and batch-invariant (`bi`) are untouched by the patch and read identically in both blocks: bf16 609.2 / 15,595.9
/ 33,607.7 and 608.1 / 16,007.5 / 33,330.1 for b1 / b32 / b128 at t0 / t1; bi 213.1 / 7,494.8 / 34,199.1 and 213.5 /
7,536.4 / 34,206.4.

**Cost of the one-hot fix: under 0.5% in every configuration, inside run-to-run noise.**

## Profile, 1B, ms per step (`out/prof_b*_t0`)

| | base b128 | repo b128 | base b32 | repo b32 | base b1 | repo b1 |
|---|---|---|---|---|---|---|
| tok/s | 6,933.0 | **9,388.4** | 4,151.0 | **5,233.1** | 214.1 | **256.9** |
| prefill step | 28.538 | **12.154** | 25.046 | **9.826** | 23.758 | **8.833** |
| decode step (median) | 10.927 | **8.332** | 5.800 | **4.473** | 4.510 | **3.844** |
| decode host sample | 1.061 | **0.699** | 0.986 | 0.707 | 0.944 | 0.658 |

The prefill step is where the fused element-wise stages pay: 28.5 → 12.2 ms at b128. Decode drops 2.6 ms at b128,
of which ~0.36 ms is host sample time and the rest is the fused kernels replacing the Inductor chain.

The same b128 config before the one-hot patch read 9,432.2 tok/s, so the fix costs **0.46%** — consistent with the e2e
blocks above.

## Microbenchmarks

Element-wise stages, µs per call (`out/micro/ew.log`), M = 1 / 8 / 32 / 128 rows: `add_norm` 11.9 / 12.1 / 12.2 /
12.6, `qkv` 2.7 / 2.7 / 2.9 / 4.6, `attn_out` 6.0 / 6.1 / 6.2 / 6.4, `silu` 32.2 / 32.7 / 32.9 / 34.3. Flat in M,
as a fused single-launch stage should be.

Fused linear vs the Inductor version (`out/micro/gemm.log`, `old → fused`, µs):

| module | M=1 | M=32 | M=128 |
|---|---|---|---|
| `0.wqkv` | 23.3 → 14.3 (1.63×) | 31.0 → 20.9 (1.48×) | 78.7 → 53.7 (1.47×) |
| `0.wo` | 16.7 → 14.2 (1.18×) | 64.7 → 15.0 (4.31×) | 209.6 → 27.7 (7.58×) |
| `0.wgu` | 54.1 → 41.3 (1.31×) | 431.1 → 63.0 (6.84×) | 446.5 → 148.0 (3.02×) |
| `0.wd` | **30.6 → 49.5 (0.62×)** | 212.3 → 50.1 (4.24×) | 350.0 → 93.5 (3.74×) |
| `lm` | 450.9 → 273.7 (1.65×) | 2,520.2 → 443.1 (5.69×) | 9,297.1 → 1,024.2 (9.08×) |

**`0.wd` at M=1 is 0.62× — slower than unfused.** Every other cell is faster. At M=1 the down-projection is a
GEMV whose launch overhead the fused version cannot amortise across rows; it is one stage of 32 in the model, so it
does not show up in the b1 e2e (254.2 vs 255.3 pre-fix), but it is a real regression in that cell and the dispatch
could keep the Inductor path for M ≤ 8 on `wd`.

## 8B, exact and int8, 3 reps, median tok/s (`out/bench8`)

| batch | exact 8B | int8 8B |
|---|---|---|
| 1 | 66.1 | 180.0 |
| 32 | 1,364.4 | 5,017.4 |
| 128 | 3,093.9 | 10,271.6 |

These are repo-tree numbers only; no 8B base run was taken for this lane, so they document the shape of the 8B
curve rather than a speedup.

## Files

`src/bitexact/vllm_plugin/exact_fuse.py` (new, 310 lines), `exact_run.py`,
`src/bitexact/vllm_plugin/{model.py,tri_logits.py}` (one-hot fix), `tests/test_vllm_exact_fuse.py`,
`tests/test_vllm_kernels.py`, `experiments/perf/p8_exactfuse/*`.