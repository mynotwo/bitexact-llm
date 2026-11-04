# speed_attn: attention-side speedups for the BitExact plugin at large batch

Host 138, H100, one GPU per run (each started at 0 MiB), vLLM 0.32, base = `exp/speed` d10ac00.
ms/step = wall of one `generate` / new tokens (prefill included), median of 3 generates (`speed.py`).

## Result

Only one change survived: **packed limb KV layout** (item 3). End-to-end gain is 5.7% at B=64 and 5.9% at B=512, under the
10% bar, so this is the last round and the work stops here.

| | base | limb KV | change |
|---|---|---|---|
| B=64, 512 tokens (ms/step) | 5.853 | 5.522 | -5.7% |
| B=512, 256 tokens (ms/step) | 26.9 | 25.31 | -5.9% |

Full test suite on 138: 83 passed, 5 skipped. Bit gate (`gate.py`, 32 GSM8K prompts x 8, T=1 and greedy, `max_num_seqs` 1 and 64): 0 of 256 sequences differ from base in
all four configurations, and B=1 == B=64 (53,387 and 65,536 tokens; tokens, log p and logits-row hash all equal).
The scorer is untouched (its int32 paged cache and the `attend` int32 path are unchanged), so the scorer gate was not rerun.

## What changed

- KV cache per token and head: K and V are integers within 3 base-256 limbs (|x| <= 8355711), stored as 3 int8 limb planes
  each (384 B, was 512 B). Declared head size 96 bf16 instead of 128, so plugin KV is 48 KB/token (was 64 KB/token) and
  vLLM sizes the cache accordingly (about 33% more tokens in the same memory).
- Attention kernels load the limb planes directly: no shift/mask split of K and V on load. `tri_paged.attend` selects the
  layout from the cache dtype (int8 limb planes, or the old int32 layout used by the scorer).
- `_store_kernel` (Triton) writes the limbs of K / V into the cache in one launch, replacing the cat + int32 scatter.
- `attend` is untouched in definition: two-pass max-then-accumulate integer softmax as before.

## Tried and dropped

- Item 1, fuse `_max_kernel` and `_acc_kernel` into one kernel (nsplit == 1 only; at B >= 64 the grid already exceeds the
  SM target so nsplit is 1). Kernel micro-benchmark (`bench_attn.py`, R=430, ctx 400, M=4): fused was slower than two
  kernels, 647 vs 571 us (int32 layout, bn 64, 4 warps) and 478 vs 453 us (limb layout). Under CUDA graphs the saved launch
  is free, and the larger register footprint of one kernel costs occupancy. Removed again, not committed.
- Warps/tile retuning: bn 64 with 2 warps is 417-501 us vs 453-571 us at 4 warps in the micro-benchmark, but earlier work
  found at most 1.07x from tuning, and it was not pursued here.
- Item 2, folding the requantise kernel into a neighbouring epilogue: not attempted. The requantise kernels
  (`triton_red_fused_*amax*`, Inductor-generated from `pre_norm` / `pre_plain`) are per-row reductions over int64 rows
  whose producers are int8-GEMM epilogues (`Segs.post`, in `int8/model.py` `_linear`, owned by the exp/speed-gemm agent)
  or `attn_rows` (int64 division); folding means a hand-written kernel for the whole elementwise chain, outside the
  write-set and with a bit-exactness risk, for a share (12% of GPU time) that could at best save a part of itself.

## Profile at B=512 (base, 64 decode steps, `speed.py --prof`)

`_acc_kernel` 18.8%, int8 GEMM (cutlass s8 128x64) 12.3%, `_max_kernel` 7.5%, then Inductor kernels (elementwise 5-7% each,
amax reductions 2.5-4.7%). Attention (`_max` + `_acc`) is 26% of GPU time; limb layout cuts its bytes by 25% and the
split ALU, which accounts for the 5.8% end-to-end figure.

## Files

`bench_attn.py` (kernel micro-benchmark), `speed.py` (`--prof` added), `gate.py`, `run138.sh` (`BS`, `GEN` env for speed
mode), `sample.py`, `compare.py`, `env138.sh`, `sync.sh` (to `~/speed_attn` on 138).
