# Llama-3.1-8B exact mode: cross-hardware results

**Conclusion.** With the public exact engine and one shared constants file (internal SHA-256
`2f02ac8e43f08ac337b397d18fc740eb48dca27158c62ef742fbd4a9e729bfa9`), every run on H800, H100, two x86
CPU hosts and an Apple M4 Pro CPU produces the same int64 logits row, greedy token and T = 0.7 sampled
token as the H800 reference at every compared position: 0 differences in 9,000+ position comparisons.
L20 is not yet compared: the first run failed at the first forward because one attention kernel exceeded sm_89 shared memory; that is fixed (`f38e1fd`, below) and the rerun, Merlin job `03a48bb8551f1cdb`, was still queued at commit time.

Workload: `workload_T0.7_8x128.json` (SHA-256 `2384dafe…f889`), 8 texts, first 128 generated positions,
595 positions; teacher forcing, so subset runs cover a prefix of the same rows. Reference:
`results/h800_141/h800_bs1.json`. Each row of the table is one `compare.py` run against it.

| Platform | Device | torch (Triton) | Mode | Positions compared | Logits rows differing | Tokens differing (greedy / sampled) |
|---|---|---|---|---|---|---|
| H800 (141) | GPU, Triton int8 | 2.4.1+cu121 (3.0.0) | bs8, incr, incr+compile | 595 each | 0 | 0 / 0 |
| H800 (141) | GPU, cuBLAS `_int_mm` | 2.4.1+cu121 (3.0.0) | bs8 | 595 | 0 | 0 / 0 |
| H800 (141) | GPU, bmm forced to 2 stages | 2.4.1+cu121 (3.0.0) | bs1, bs8, incr | 595 each | 0 | 0 / 0 |
| H100 (138) | GPU, Triton int8 | 2.6.0+cu124 (3.2.0) | bs1, bs8, incr, incr+compile | 595 each | 0 | 0 / 0 |
| x86 CPU, Xeon 8457C (138) | CPU float64 | 2.6.0+cu124 | bs1, bs8, incr | 595 each | 0 | 0 / 0 |
| x86 CPU, Xeon 8457C (141) | CPU float64 | 2.4.1+cu121 | bs1, bs8 | 595 each | 0 | 0 / 0 |
| x86 CPU, Xeon 8457C (141) | CPU float64 | 2.4.1+cu121 | incr (`--positions 16`) | 113 | 0 | 0 / 0 |
| x86 CPU, Xeon 8457C (141) | CPU float64, `--cpu-lazy` | 2.4.1+cu121 | bs1 (3 texts; 1 text × 4) | 176; 3 | 0 | 0 / 0 |
| Apple M4 Pro (arm64, macOS 26.3.2) | CPU float64, `--cpu-lazy` | 2.15.0 | bs1, bs8 | 595 each | 0 | 0 / 0 |
| Apple M4 Pro | CPU float64, `--cpu-lazy` | 2.15.0 | incr (`--positions 32`) | 198 | 0 | 0 / 0 |
| L20 (Merlin, sm_89) | GPU, Triton int8 | 2.4.1 (3.0.0) | bs1, bs8, incr, incr+compile | pending (job `03a48bb8551f1cdb`) | not verified | not verified |

## Checks

- Constants. The internal SHA-256 was recomputed from the tensors (not only read from the file's label)
  on 141 (`const_8b_public.pt`, built with the public code), on 138 (`~/a18/out8b/const_8b.pt`, the A18.2
  build) and on the M4 (copy of 141's file, file SHA-256 `c0890062…a8c9` equal on both ends): all
  `2f02ac8e…`. The L20 job builds its own constants from the 4 shards (shard SHA-256 checked) and checks the
  same digest; it matched (`all_same_bits` true).
- Prefixes. For every result file, the set of keys equals exactly the first `positions` generated tokens
  of the first `texts` samples of the workload, and its workload SHA-256 equals the reference's.
- Comparison. `compare.py` compares the full 64-hex SHA-256 of each int64 logits row plus both tokens and
  exits 1 on any difference; a file with one hash digit changed is reported as 1 differing row.

## Code changes, for review

- Port to 8B (`c592a62`): geometry read from the constants' config; sharded checkpoints; RMSNorm's 1/D as
  a shift for any power-of-two D ≤ 4096 (identical integers for D = 2048: same `m` and shift); attention
  scale for head_dim 128 as one rounded constant `round(log2(e)/sqrt(2) · 2^28) = 273841598`
  (273841597.53 checked at 60 digits); `cpu_lazy` builds the same float64 weights per call. No change in
  the arithmetic for Llama-3.2-1B: 256 positions of `bitexact.repro record` identical before and after.
- L20 fix (`f38e1fd`): at head_dim 128 the attention `bmm` tile (BM 64, BN 128, BK 64) needs 110,592 B of
  shared memory with 3 pipeline stages; sm_89 allows 101,376 B, so every L20 run failed at the first
  forward (`OutOfResources`). `bmm` now retries with 2 stages (61,440 B) and remembers the choice per
  device and tile. Stages only change prefetching; int32 limb sums are exact, so the integers are the same
  (shown on H800 with every tile forced to 2 stages, row 3 of the table; `tests/test_bmm_stages.py`).
  Kernel and sizes found by compiling all 8B kernel shapes for an sm_89 target on an H800. The weight GEMM
  (`wmm`, 71,680 B) fits unchanged.

## Not verified

- Heterogeneity: H800 and H100 are both sm_90 (same Triton kernels, different torch/Triton/host); the two
  x86 hosts have the same CPU model. The distinct arithmetic paths compared are Triton int8 tensor cores,
  cuBLAS int8, x86 float64 BLAS and arm64 float64 BLAS; sm_89 (L20) would add a different GPU architecture once its rerun completes.
- M4 incremental decoding covered 32 positions per text (198 rows), not 128; 141 CPU incr 16 positions.
- incr+compile and cuBLAS ran only on GPUs listed above; CPU runs never use `torch.compile`.
- Workload is 8 texts at one temperature; longer contexts (> ~700 tokens) are not covered.
