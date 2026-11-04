# Faster int8 scorer, same bits

Branch `exp/speed-scorer` (from `exp/profile`). Scorer = teacher-forced log p of every response token over prompt + response, from the int8 engine's integer logits. One H100 80GB (host 138), torch 2.13.0+cu130, Triton 3.7.1. Nothing in `src/bitexact/vllm_plugin/` was edited; `tri_paged.attend` and `weight_table` are imported read-only.

## Summary

| tok/s (median of 3) | GSM8K b8 | 4k b1 | 4k b4 |
|---|---|---|---|
| eager reference (`compiled=False`, `fused=False`, as abc_measure ran it) | 6,637 | 2,361 | 3,658 |
| a. compiled segments by default | 27,239 | 9,391 | 12,922 |
| b. + attention through `tri_paged.attend` | 31,903 | 11,573 | 19,434 |
| c+d. + packed varlen scorer, no per-chunk host syncs | 36,763 | 20,140 | 20,838 |
| speed-up vs eager | 5.5x | 8.5x | 5.7x |
| speed-up vs compiled (a) | 1.35x | 2.14x | 1.61x |

GSM8K = the 1,024 plugin rollouts `q1_B.json` (430,739 tokens, batch 8). 4k = the 16 plugin rollouts `q2_4096_B.json` (68,947 tokens). Timing counts prompt + response tokens, the float64 log-softmax and the copy of the log p to the host; it does not hash logits. One warm-up pass precedes the 3 timed passes, so compile time is excluded. Each item was measured on an otherwise idle GPU of 138 (other jobs ran on other GPUs; host CPU load was not controlled).

## Bit-exactness gate (every item)

Per response row, `sha256(int64 logits row)[:16]` equals (1) the hash the plugin recorded in the rollout file and (2) the hash from the eager reference run saved before any change; the float64 log p arrays are compared bit for bit (`view(int64)`) with the eager reference. Eight configurations per item: GSM8K first 128 sequences (28,083 rows) at batch 8, 64, 1 and batch 8 with a shuffled sequence order; 4k all 16 sequences (65,536 rows) at batch 4, 1, 8 and batch 4 shuffled. Items a, b, c: 8 of 8 configurations equal on all three comparisons.

| item | configurations equal | rows equal to plugin hash |
|---|---|---|
| a | 8 / 8 | 28,083 / 28,083 and 65,536 / 65,536 |
| b | 8 / 8 | same |
| c+d | 8 / 8 | same |
| final tree (commit of the push) | 2 / 2 run (GSM8K b8 shuffled, 4k b4); all 1,024 GSM8K sequences (215,899 rows) equal the plugin hashes | 215,899 / 215,899 |

CPU: the CPU path is untouched (`compiled` and `fused` default to off on CPU; `Scorer.score` runs the padded reference flow there). On 16 GSM8K sequences (3,501 rows, batch 4) the CPU float64 path gives rows equal to the plugin hashes: 16 / 16 sequences, 3,501 / 3,501 rows equal (final tree).

## Per item

**a. Compiled by default** (`IntLlama(compiled=None)` = on for CUDA, off for CPU; `compiled=False` is the eager reference). 6,637 -> 27,239 tok/s on GSM8K b8 (4.1x), 2,361 -> 9,391 on 4k b1 (4.0x), 3,658 -> 12,922 on 4k b4 (3.5x). Bit check passed.

**b. Fused causal attention** (`IntLlama(fused=None)` = on for CUDA). The scorer's K/V live in `PagedCache` (the plugin's `[blocks, HKV, bs, 2 HD]` int32 layout, one run of blocks per sequence) and attention is `tri_paged.attend`: two sweeps, no int64 score tensor, keys past each 64-query tile's last position are not read. 27,239 -> 31,903 (1.17x), 9,391 -> 11,573 (1.23x), 12,922 -> 19,434 (1.50x). Bit check passed.

**c. Packed scorer** (`int8/scorer.py`, `Scorer`). A batch becomes one flat stream of real tokens (no padding rows). Chunks of about 2,048 tokens, rounded up to a multiple of 128 rows so the compiled segments see few distinct shapes (compiled with `dynamic=False`: shape-specialised kernels, which were faster than the shape-generic ones). Cache slots, gather/scatter indices, query tiles (64 tokens, tail in 16-token tiles) and block-table bases are built in numpy and uploaded in one copy per batch. Only the rows that predict a response token leave the last layer. 31,903 -> 36,763 (GSM8K, +15%), 11,573 -> 20,140 (4k b1, +74%), 19,434 -> 20,838 (4k b4, +7%). The 4k b1 gain is mostly fewer, larger launches: the padded flow ran 256-row forwards at batch 1.

**d. Host syncs and CUDA graphs.** The per-chunk `.item()` of the log p is gone: log p of all rows of a batch is accumulated on the device and copied once per batch, and `m.check()` is the only other sync. This is part of the c measurement, not measured alone. CUDA graphs were not implemented. Reason: after c the GPU is idle 9.2% of the wall on GSM8K b8, 4.1% on 4k b1 and 3.4% on 4k b4 (`prof.py`, torch.profiler, union of kernel intervals), so graphs could give at most 1.10x / 1.04x / 1.04x; a graph needs fixed tile counts and block-table widths per chunk, and these differ for almost every chunk of real batches, so it would need tile padding and many captures for under 10%. Not done.

**e. int32 storage between segments: tried, no gain, reverted.** Residual, projections, gate/up, q and the attention output stored as int32 (widened to int64 inside the compiled segments, with the bound flag raised when a value would not fit), bit-exact on 32 GSM8K sequences. Speed on the same subset against the committed packed scorer: GSM8K first 256 sequences 35,807 -> 35,594 tok/s, 4k b1 20,021 -> 19,656, 4k b4 20,817 -> 20,512. The gate "faster than the previous item" fails, so the change is not in the tree; the patch is `i32_attempt.patch`. The compiled int64 segments already fuse the widening and narrowing, so the saved bytes are small next to the GEMM, attention and limb work. Raw scores (38 bit), num (46 bit) and den stay int64, as planned.

Attention launch parameters of `tri_paged.attend` (not changed): 8 warps and BN = 128 were both slower on 4k b1 (second timed pass 4.24 s and 4.35 s against 3.42 s with the defaults).

## Where the packed scorer spends time (profile, GPU kernel time over wall)

| workload | tok/s in profile | GPU idle | top kernels |
|---|---|---|---|
| GSM8K b8 | 34,780 | 9.2% | int8 GEMM 128x64 17.5%, Inductor limb-split/epilogue 15.6%, `_acc_kernel` 14.6%, ... |
| 4k b1 | 19,780 | 4.1% | `_acc_kernel` 40.3%, int8 GEMM 11.5%, ... |
| 4k b4 | 20,634 | 3.4% | `_acc_kernel` 41.0%, ... |

At 4k the attention accumulate kernel is the largest item, so further gains there belong to `tri_paged` (plugin side, see below).

## Cross-GPU check, final scorer on an H800 of host 141

Host 141 (driver 535.183.06, CUDA 12.2 driver, torch 2.13.0+cu129, Triton 3.7.1, GPU 3 with 0 MiB used) runs the env. The final scorer, with hashes and log p compared against the eager reference that was recorded on the H100 of 138 and the plugin hashes:

| config on H800 | rows equal plugin hash | sequences equal ref hash / log p bits |
|---|---|---|
| GSM8K 128 seq, batch 8 | 28,083 / 28,083 | 128 / 128 and 128 / 128 |
| GSM8K 128 seq, batch 64, shuffled | 28,083 / 28,083 | 128 / 128 and 128 / 128 |
| 4k 16 seq, batch 4 | 65,536 / 65,536 | 16 / 16 and 16 / 16 |
| 4k 16 seq, batch 1 | 65,536 / 65,536 | 16 / 16 and 16 / 16 |

H800 speed (shared host, other jobs on other GPUs): 4k b1 18,344 tok/s (median of 3), GSM8K b8 first 256 sequences 31,109 tok/s (first pass 22.7 s includes compile). So the scorer's integers and log p are the same on H100 and H800.

## Other measurements

- Chunk size (`rows`), GSM8K b8, first 256 sequences: 1,024 rows 32,694 tok/s, 2,048 rows 35,807 (default). The padded flow with fused attention (item b) on the same subset: 31,920.
- Tests: `tests/test_scorer.py` (new: packed scorer equals the eager padded reference bit for bit for chunk sizes 64, 256, 2,048, shuffled order and alone; CPU scorer runs the reference flow), `tests/test_invariance.py` (now also runs eager unfused and compiled unfused int8 variants). `tests/test_gpu_kernels.py` unchanged. On 138: 6 passed, 3 skipped (the exact-engine parametrisation), `ruff check` clean.

## Left undone / needs the plugin side

- CUDA graphs (see d).
- `tri_paged.attend` needs nothing to be used here. Two changes there would help: (1) the `_acc_kernel` is 40% of the 4k scorer; the second QK sweep it repeats is the cost the profile report named as rank 6 (single online-softmax sweep, which changes the integer softmax and needs new reference numbers); (2) `attend` allocates `mx`, `num`, `den` with `torch.full/zeros` on every call and the two kernels use atomics even when `nsplit` is 1, which a no-split path could skip. Not edited.
- Per-sequence `lm_head` is computed on chunks of 512 stream rows padded to a multiple of 128; the log-softmax is float64 over 128,256 columns and takes 2 to 3% of the time.
- The first call with a new batch shape compiles (tens of seconds); the cache of compiled shapes is bounded by 128-row buckets.

## Files

`src/bitexact/int8/scorer.py` (packed scorer, reference padded flow), `int8/model.py` (`compiled`, `fused`, `dynamic` options, `PagedCache`), `int8/segs.py` (`qkv_rows`, `attn_rows`, shape mode), `tests/test_scorer.py`. Harness: `bench.py` (speed and bit check; `--mode old|packed`), `prof.py`, `run_item.sh`, `sync.sh`, `env138.sh`, `show.py`, `i32_attempt.patch`.
