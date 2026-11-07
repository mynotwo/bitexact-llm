# p18: byte-aligned and 4-aligned block grids for exact-mode weights

A measurement, not a code change. p17 found that a per-block grid cuts mean int8 limbs per weight from 3.00 to
2.21 at block 32, but in the direct formulation each output element then needs K/B shift-and-add steps. The
variant that avoids those steps restricts each block's grid exponent to a multiple of 8, so a block's
contribution lands on a different limb diagonal instead of needing its own shift class. This study measures
what that restriction costs in limbs, and also a multiple-of-4 grid.

## Headline

Byte-aligned (multiple of 8) grids keep little of the reduction. At block 32 the mean is 2.85 limbs on 1B
(8.55 products, against 6.63 for the free grid and 9.00 for the row grid). Mean (bot mod 8) is 4.6 at block 32,
above the uniform-residue 3.5, so the alignment penalty is worse than uniform at the block size that matters.
A multiple-of-4 grid keeps more: 2.74 limbs at block 32 (8.21 products) on 1B.

Surprising side result, now measured on all layers of both models: the free grid needs few exponents per row.
At block 32 a row needs 2.6 distinct grid exponents on 1B and 2.4 on 8B (max 8), not K/B. p17's accumulation
estimate assumed K/B. This measurement refutes that assumption. Section "Accumulator classes and int32 headroom"
gives the accumulator count and its int32 bound.

## Method

Same reader and same shipped `bf16_parts`, `_bitlen`, `_lowbit` as p17 (`experiments/perf/p18_bytegrid/analyze.py`,
a copy of p17's with the aligned-grid counts added). Per block, top = max(ex + bitlen(mag)), bot = min(ex +
lowbit(mag)) over nonzero entries, span = top - bot. Signed base-256 capacities C = 7, 15, 23 bits for 1, 2, 3 limbs.

Three grids:

- free (p17): g = top - C, exact at C when every entry has ex + lowbit >= g.
- 8-aligned: g = 8 floor(bot / 8). Exact at C iff top - g <= C, equivalently span + (bot mod 8) <= C.
- 4-aligned: g = 4 floor(bot / 4), exact at C iff top - g <= C.

For each block the chosen L is the smallest of {1, 2, 3} whose residual count is within the budget. Two rules are
reported. The literal p17 rule (budget tau = 1e-4 of the block's nonzeros) is the one compared against p17. The
exact rule (tau = 0, zero residual) is the one that matches "exact". The table below uses the literal rule;
zero-residual numbers follow.

Checks. `check_grid.py` compares `block_arrays()` with a plain-Python brute force on a random 24 x 96 bf16 matrix
at blocks 32, 16, 8, and whole rows: counts, spans and the bot-mod-8 histogram all match, and the closed form
"zero residual on the 8-aligned grid iff span + (bot mod 8) <= C" holds for every block. The free-grid means
reproduce p17 (1B block 32: 2.212; row: 3.000).

Coverage: 1B is all 16 layers and all 7 linear types, and lm_head is reported separately. 8B covers layers 0, 15
and 31 only (sample, not the full 32 layers), with all 7 linear types and lm_head.

## Mean limbs and products, literal rule (tau = 1e-4)

Mean limbs per weight, element-weighted over all linear layers (lm_head excluded). Products = 3 x mean limbs.

| Block | free 1B | 4-aligned 1B | 8-aligned 1B | free 8B (L0,15,31) | 4-aligned 8B | 8-aligned 8B |
|---|---:|---:|---:|---:|---:|---:|
| row | 3.000 (9.00) | 3.000 (9.00) | 3.000 (9.00) | 3.000 (9.00) | 3.000 (9.00) | 3.000 (9.00) |
| 256 | 2.893 (8.68) | 2.998 (8.99) | 3.000 (9.00) | 2.920 (8.76) | 2.985 (8.95) | 3.000 (9.00) |
| 128 | 2.663 (7.99) | 2.988 (8.96) | 2.998 (8.99) | 2.699 (8.10) | 2.911 (8.73) | 3.000 (9.00) |
| 64 | 2.404 (7.21) | 2.931 (8.79) | 2.971 (8.91) | 2.429 (7.29) | 2.750 (8.25) | 2.992 (8.98) |
| 32 | 2.212 (6.63) | 2.737 (8.21) | 2.850 (8.55) | 2.231 (6.69) | 2.516 (7.55) | 2.927 (8.78) |
| 16 | 2.099 (6.30) | 2.426 (7.28) | 2.624 (7.87) | 2.115 (6.35) | 2.279 (6.84) | 2.747 (8.24) |
| 8 | 2.042 (6.13) | 2.182 (6.54) | 2.392 (7.18) | 2.052 (6.16) | 2.124 (6.37) | 2.509 (7.53) |

Fraction of blocks needing L = 1, 2, 3, and fraction still over 23 bits (no grid holds them even at L = 3), at
block 32, literal rule:

| Model | Grid | L=1 | L=2 | L=3 | over 23 bits |
|---|---|---:|---:|---:|---:|
| 1B | free | 0.000 | 0.788 | 0.212 | 0.0009 |
| 1B | 4-aligned | 0.000 | 0.263 | 0.737 | 0.0062 |
| 1B | 8-aligned | 0.000 | 0.150 | 0.850 | 0.0076 |
| 8B (L0,15,31) | free | 0.000 | 0.769 | 0.231 | 0.0013 |
| 8B (L0,15,31) | 4-aligned | 0.000 | 0.484 | 0.516 | 0.0064 |
| 8B (L0,15,31) | 8-aligned | 0.000 | 0.073 | 0.927 | 0.0129 |

No block fits in one limb under any grid (the 1-limb row is 0.000 everywhere), matching p17's 8-bit significand
versus 7-bit limb.

Residual density at the chosen L (fraction of nonzero entries that the chosen grid cannot hold, literal rule):
1B block 32 is 2.9e-5 (free), 1.9e-4 (4-aligned), 2.4e-4 (8-aligned). 8B block 32 is 4.1e-5 (free), 2.0e-4
(4-aligned), 4.1e-4 (8-aligned). The p17 free-grid density of about 5e-5 at row level is reproduced at 1B row (5.4e-5).

Zero-residual (exact rule, tau = 0): mean limbs are identical to the literal rule to three decimals at every block
size and grid (for example 1B block 32: free 2.212, 4-aligned 2.737, 8-aligned 2.850). The budget does not bind at
these block sizes, as in p17. The fraction of blocks over 23 bits at zero residual is larger: at row size 12% (1B)
and 25% (8B) for the free grid, and 42% (1B) and 75% (8B) for the 8-aligned grid. At 256 size the 8-aligned grid is
still 6% (1B) and 9% (8B) over 23 bits, so it cannot be used at row or 256 size without a fourth limb.

## Bot mod 8 distribution

The alignment penalty is (bot mod 8) bits on top of the span, so its distribution decides the cost. Over nonempty
blocks, fraction with bot mod 8 = 0..7, all linear layers, 1B:

| Block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| row | 0.221 | 0.217 | 0.129 | 0.048 | 0.038 | 0.065 | 0.112 | 0.171 | 2.96 |
| 256 | 0.056 | 0.102 | 0.168 | 0.233 | 0.233 | 0.134 | 0.043 | 0.031 | 3.21 |
| 64 | 0.042 | 0.031 | 0.056 | 0.102 | 0.169 | 0.234 | 0.233 | 0.133 | 4.62 |
| 32 | 0.131 | 0.040 | 0.031 | 0.056 | 0.102 | 0.169 | 0.235 | 0.234 | 4.58 |
| 8 | 0.243 | 0.238 | 0.121 | 0.034 | 0.031 | 0.057 | 0.104 | 0.174 | 2.82 |

8B (layers 0, 15, 31) at block 32 is 0.078, 0.031, 0.045, 0.079, 0.132, 0.199, 0.243, 0.193, mean 4.69.

Headline: the penalty is not the uniform 3.5 bits. At blocks 32 and 64 the mean is about 4.6 bits, worse than
uniform, because the residue sits near 6 and 7. At block 8 it is about 2.8, better than uniform. The byte-aligned
penalty therefore depends on the block size and does not average out.

## Distinct grid exponents per output row (free grid, all layers)

Free grid: g = top - 23 per block (3 limbs, exact for every block). For each output row, count distinct g over its
nonempty blocks, and distinct (g mod 8). All 16 layers of 1B and all 32 layers of 8B, all 7 linear types; lm_head
separate (1B ties lm_head to embed_tokens, so it is the embedding tensor). Counts are exact (histograms kept per
tensor, percentiles from them). `free_rows.py` checks the vectorized counts against a plain-Python loop
(`--check`, zero mismatches). Rows: 376,832 (1B), 1,376,256 (8B).

Distinct g per row, all linear layers:

| Block | 1B mean | p50 | p90 | p99 | max | 8B mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.65 | 2 | 2 | 3 | 6 | 1.67 | 2 | 2 | 3 | 6 |
| 128 | 1.79 | 2 | 2 | 3 | 6 | 1.97 | 2 | 2 | 3 | 7 |
| 64 | 2.11 | 2 | 3 | 4 | 6 | 2.16 | 2 | 3 | 4 | 7 |
| 32 | 2.57 | 3 | 3 | 4 | 7 | 2.40 | 2 | 3 | 4 | 8 |
| 16 | 2.94 | 3 | 4 | 4 | 7 | 3.07 | 3 | 4 | 5 | 8 |
| 8 | 3.78 | 4 | 4 | 5 | 8 | 3.87 | 4 | 4 | 6 | 10 |

Distribution of rows by number of distinct g (fraction of rows; 13 or more: 0 at every block size):

| Block | Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1B | 0.373 | 0.606 | 0.019 | 0.001 | 0.00005 | 0 | 0 | 0 | 0 |
| 256 | 8B | 0.372 | 0.592 | 0.033 | 0.003 | 0.0002 | 0.00001 | 0 | 0 | 0 |
| 128 | 1B | 0.269 | 0.673 | 0.054 | 0.004 | 0.0002 | 0.00001 | 0 | 0 | 0 |
| 128 | 8B | 0.111 | 0.814 | 0.068 | 0.006 | 0.0005 | 0.00003 | 0 | 0 | 0 |
| 64 | 1B | 0.112 | 0.677 | 0.199 | 0.012 | 0.001 | 0.00003 | 0 | 0 | 0 |
| 64 | 8B | 0.007 | 0.843 | 0.136 | 0.011 | 0.002 | 0.0001 | 0 | 0 | 0 |
| 32 | 1B | 0.002 | 0.460 | 0.508 | 0.027 | 0.003 | 0.0002 | 0 | 0 | 0 |
| 32 | 8B | 0 | 0.636 | 0.331 | 0.029 | 0.004 | 0.0003 | 0.00002 | 0 | 0 |
| 16 | 1B | 0 | 0.211 | 0.652 | 0.127 | 0.009 | 0.001 | 0.00004 | 0 | 0 |
| 16 | 8B | 0 | 0.060 | 0.828 | 0.100 | 0.011 | 0.002 | 0.0002 | 0.00001 | 0 |
| 8 | 1B | 0 | 0.0004 | 0.318 | 0.592 | 0.082 | 0.007 | 0.001 | 0.00005 | 0 |
| 8 | 8B | 0 | 0 | 0.231 | 0.681 | 0.076 | 0.010 | 0.002 | 0.0003 | 0.00007 |

Collapse modulo 8. The exact sum over rows of (distinct g minus distinct (g mod 8)) is 0 for every block size on
both models, except 8B block 8, where it is 177 over 1.38M rows. Exponents in one row almost never differ by 8 or
more, so the mod-8 count equals the g count. Distinct (g mod 8) therefore does not reduce the accumulator count
below what the g count gives.

## Accumulator classes and int32 headroom

Accumulator count per output element = 5 x distinct (g mod 8), since each class carries the 5 limb diagonals.
Per row, the K positions in one class are the K positions of blocks whose g mod 8 equals that residue. The worst
class is the largest such count over all rows. Its partial sum is bounded by that count times 3 diagonal products
per k times 128 x 128, the largest product of two signed int8 limbs. Using the coordinator's form (count x 127 x
127, no diagonal factor) gives a bound 3x smaller.

| Block | 1B acc mean (p99, max) | 8B acc mean (p99, max) | max K in one class 1B / 8B | int32 headroom 1B / 8B |
|---|---|---|---|---|
| 256 | 8.2 (15, 30) | 8.3 (15, 30) | 8192 / 14336 | 5.3x / 3.0x |
| 128 | 9.0 (15, 30) | 9.8 (15, 35) | 8192 / 14336 | 5.3x / 3.0x |
| 64 | 10.6 (20, 30) | 10.8 (20, 35) | 8192 / 14016 | 5.3x / 3.1x |
| 32 | 12.9 (20, 35) | 12.0 (20, 40) | 8000 / 13472 | 5.4x / 3.2x |
| 16 | 14.7 (20, 35) | 15.3 (25, 40) | 7264 / 12768 | 6.1x / 3.5x |
| 8 | 18.9 (25, 40) | 19.4 (30, 40) | 6256 / 11336 | 7.1x / 3.9x |

Headroom = 2^31 / (max K in one class x 3 x 16384). The minimum is 3.0x (8B, K = 14336, all K in one class).
The coordinator's 127 x 127 form gives 9.3x on 8B. No block size or model overflows int32, even in the worst row.

The worst class is the whole K (8192 on 1B, 14336 on 8B), so the bound is set by the largest K, not by the number
of classes. The number of classes sets how many accumulators a tile holds, not the overflow risk.

## Type and layer depth

Mean distinct g at block 32 by type, all layers:

| Model | q | k | v | o | gate | up | down |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1B | 2.46 | 2.59 | 2.48 | 2.97 | 2.47 | 2.49 | 3.02 |
| 8B | 2.68 | 2.77 | 2.63 | 2.89 | 2.30 | 2.22 | 2.49 |

Mean distinct g at block 32 by layer depth: 1B ranges 2.42 to 2.75, lowest in the middle layers and highest at
layers 0 and 15 (2.69 and 2.75). 8B ranges 2.29 to 2.60, highest at layer 0 (2.55) and layer 31 (2.60), with
layers 1 to 30 between 2.29 and 2.48. Type and depth move the mean by under 1 exponent, with no tail: the maximum
per row stays at 8 for block 32 on both models.

## Free grid per 2D tile (N output rows x K weights)

Same free grid (g = top - C), same literal rule (tau = 1e-4 of the tile's nonzeros), but the grid is chosen per tile:
top and bot are taken over all N x K elements of the tile, so every row in the tile shares one grid. Shape 1 x K is
the 1D per-row-block case from earlier sections and reproduces them (1B 1x32: 6.63 products, 2.57 distinct g,
12.9 accumulators). Mean limbs are element-weighted; tiles of all-zero elements take L = 1, as in p17. All 16 layers of
1B and all 32 layers of 8B, linear types only; lm_head separate in the JSON (1B 1x32 lm_head: 6.62).
`tile_grid.py` checks the vectorized counts against a plain-Python loop on a 128 x 128 matrix (zero mismatches).

Products (3 x mean limbs), L2 / L3 fraction of tiles, fraction of tiles still over 23 bits at L = 3, residual
density at the chosen L, side data in bits per weight (8 / (N x K)):

| Tile N x K | 1B products | 1B L2 / L3 | 1B over 23 | 1B resid. dens. | 8B products | 8B L2 / L3 | 8B over 23 | 8B resid. dens. | side bits/w |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| 1 x 32 | 6.63 | 0.788 / 0.212 | 0.0009 | 2.90e-5 | 6.62 | 0.794 / 0.206 | 0.0009 | 2.92e-5 | 0.2500 |
| 1 x 64 | 7.21 | 0.596 / 0.404 | 0.0020 | 3.19e-5 | 7.21 | 0.596 / 0.404 | 0.0021 | 3.36e-5 | 0.1250 |
| 1 x 128 | 7.99 | 0.337 / 0.663 | 0.0044 | 3.44e-5 | 8.05 | 0.317 / 0.683 | 0.0049 | 3.81e-5 | 0.0625 |
| 16 x 32 | 8.98 | 0.008 / 0.992 | 0.0235 | 4.66e-5 | 8.98 | 0.005 / 0.995 | 0.0234 | 4.67e-5 | 0.0156 |
| 32 x 32 | 9.00 | 0 / 1 | 0.0532 | 5.37e-5 | 9.00 | 0 / 1 | 0.0491 | 5.04e-5 | 0.0078 |
| 32 x 64 | 9.00 | 0 / 1 | 0.1150 | 6.05e-5 | 9.00 | 0 / 1 | 0.1008 | 5.42e-5 | 0.0039 |
| 64 x 32 | 9.00 | 0 / 1 | 0.1165 | 6.15e-5 | 9.00 | 0 / 1 | 0.1031 | 5.55e-5 | 0.0039 |
| 64 x 64 | 9.00 | 0 / 1 | 0.2397 | 6.89e-5 | 9.00 | 0 / 1 | 0.2058 | 6.05e-5 | 0.0020 |
| 64 x 128 | 9.00 | 0 / 1 | 0.4434 | 7.58e-5 | 9.00 | 0 / 1 | 0.3850 | 6.64e-5 | 0.0010 |
| 128 x 32 | 9.00 | 0 / 1 | 0.2442 | 7.06e-5 | 9.00 | 0 / 1 | 0.2076 | 6.08e-5 | 0.0020 |
| 128 x 64 | 9.00 | 0 / 1 | 0.4528 | 7.81e-5 | 9.00 | 0 / 1 | 0.3884 | 6.69e-5 | 0.0010 |
| 128 x 128 | 9.00 | 0 / 1 | 0.3834 | 8.58e-5 | 9.00 | 0 / 1 | 0.3040 | 7.46e-5 | 0.0005 |

Accumulators per tile-row-group. A tile-row-group is the N rows of one row-block, over all K-tiles along K. Distinct g
and distinct (g mod 8) are counted over those K-tiles (they were equal in every case here, as in p18b). Accumulators
= 5 x distinct (g mod 8). Max K in one class is the largest total K-span sharing one (g mod 8) class over all
row-groups. Headroom = 2^31 / (max K in one class x 3 x 128 x 128).

| Tile N x K | 1B distinct g mean (p99, max) | 1B accumulators mean (p99, max) | 1B max K / class, headroom | 8B distinct g mean (p99, max) | 8B accumulators mean (p99, max) | 8B max K / class, headroom |
|---|---|---|---|---|---|---|
| 1 x 32 | 2.57 (4, 7) | 12.9 (20, 35) | 8000, 5.5x | 2.40 (4, 8) | 12.0 (20, 40) | 13472, 3.2x |
| 1 x 64 | 2.11 (4, 6) | 10.6 (20, 30) | 8192, 5.3x | 2.16 (4, 7) | 10.8 (20, 35) | 14016, 3.1x |
| 1 x 128 | 1.79 (3, 6) | 9.0 (15, 30) | 8192, 5.3x | 1.97 (3, 7) | 9.8 (15, 35) | 14336, 3.05x |
| 16 x 32 | 2.20 (4, 5) | 11.0 (20, 25) | 7744, 5.6x | 2.18 (4, 6) | 10.9 (20, 30) | 14336, 3.05x |
| 32 x 32 | 2.22 (4, 5) | 11.1 (20, 25) | 7648, 5.7x | 2.14 (4, 6) | 10.7 (20, 30) | 14336, 3.05x |
| 32 x 64 | 2.10 (4, 5) | 10.5 (20, 25) | 8064, 5.4x | 2.07 (4, 5) | 10.4 (20, 25) | 14336, 3.05x |
| 64 x 32 | 2.33 (4, 5) | 11.6 (20, 25) | 7968, 5.5x | 2.34 (4, 5) | 11.7 (20, 25) | 14336, 3.05x |
| 64 x 64 | 2.17 (4, 5) | 10.8 (20, 25) | 8192, 5.3x | 2.23 (4, 5) | 11.2 (20, 25) | 14336, 3.05x |
| 64 x 128 | 1.87 (4, 5) | 9.3 (20, 25) | 8192, 5.3x | 2.08 (4, 5) | 10.4 (20, 25) | 14336, 3.05x |
| 128 x 32 | 2.46 (4, 5) | 12.3 (20, 25) | 8128, 5.4x | 2.53 (4, 5) | 12.6 (20, 25) | 14336, 3.05x |
| 128 x 64 | 2.16 (4, 5) | 10.8 (20, 25) | 8192, 5.3x | 2.37 (4, 5) | 11.9 (20, 25) | 14336, 3.05x |
| 128 x 128 | 1.89 (4, 5) | 9.4 (20, 25) | 8192, 5.3x | 2.20 (4, 5) | 11.0 (20, 25) | 14336, 3.05x |

Reading the tables. (1) The tile does not lower the grid count. Every N x K tile still needs 1.9 to 2.5 distinct
exponents per tile-row-group, and the count does not fall as N grows. The exponent variation is along K: each K-tile
has its own top, and the N rows in a tile do not share one. Sharing a grid across rows removes no K-tile grids.
(2) The tile raises the cost of the 23-bit window. With N rows sharing one top, the top is set by the widest row of
the tile. For N of 16 or more, a 23-bit window fails for 2 to 45% of tiles, so L = 3 is forced and the residual
exceeds the budget there. The product count sits at 9.00 for every tile of N of 32 or more. (3) At N = 16 the
product count is 8.98, which is not a meaningful reduction.

Verdict on the 2D question. No tile shape gives products meaningfully below 9 while the accumulator count stays
near 5. Tiles with N of 32 or more give 9.00 products (no reduction). The only shape below 9 is 16 x 32 at 8.98
products, a 0.2% reduction, with 11 accumulators per tile-row-group on average and max 25 (1B) or 30 (8B). The
shapes that reduce products (1 x K) do so only with one row per grid, which keeps 9 to 13 accumulators per output
element. The coordinator's premise that a shared 2D grid keeps accumulators at 5 is refuted by the distinct-g counts.

## Verdict

Three questions, answered separately.

(a) Byte-aligned and 4-aligned block grids are not worth a kernel. At block 32 they reach 8.55 and 8.21 products on
1B (free grid 6.63) and 8.78 and 7.55 on 8B (free grid 6.69). Their stated purpose was to avoid extra shift
classes, but the measured free-grid count is about 2.4 exponents per row, so that purpose is weak.

(b) The free grid per block (one row, one block of K) with per-exponent accumulators is the only layout measured that
gets products below 9 meaningfully (6.63 at block 32). Measured: 12 to 13 accumulators per output element at block 32
on average (5 x distinct (g mod 8)), p99 20, max 40, and int32 headroom of at least 3.05x in the worst row of 8B.
The remaining unknown is register pressure: a tile of output elements that holds 12 to 40 int32 accumulators per
element, against 5 today, may not fit the register file at the tile shapes a kernel would use. Only a timed kernel
can settle that.

(c) 2D tiles (N output rows sharing one grid) do not help. The best shape, 16 x 32, reaches 8.98 products, and every
tile with N of 32 or more is at 9.00. Accumulator counts stay at 10 to 13 mean because the grid varies along K, not
across rows. So the 2D tile is not a route to the 5-accumulator layout.

Next step: time a per-exponent-class accumulator kernel on the free 1 x 32 layout. That is a separate study.

## What this supports and what it does not

Supports: a shared grid across a 2D tile does not reduce the grid count or the accumulator count, and N of 32 or more
leaves products at 9.00; 8-aligned grids recover little (8.55 products at block 32 on 1B, 8.78 on 8B); 4-aligned grids recover more
(8.21, 7.55) at one extra shift class; the 1-limb case is impossible for any grid; the free grid needs about 2.4 to
2.6 distinct exponents per row at block 32, against K/B; the mean varies by about one exponent across type and depth; the int32 bound holds on the free grid for both models.

Does not support: any throughput claim, since these are storage counts and bounds, and no kernel was written or
timed; register pressure, which is the open question; any full-model 8B limb result (only layers 0, 15, 31 were run
for the limb tables; the distinct-exponent counts cover all 32 layers); any claim about the activation side, whose
3 limbs come from the residual-stream grid.

## Caveats carried from p17

These are storage counts, not kernel time. The 1-limb case is impossible because a bf16 significand is 8 bits and a
signed int8 limb holds 7. The literal rule uses tau = 1e-4 of the block's nonzeros; the zero-residual exact rule is
reported above and gives the same qualitative answer. The residual is stored separately and is not counted in the
limb means except through the chosen L.

## Reproduction

CPU only, `CUDA_VISIBLE_DEVICES=""`, no GPU. Host `audreymao@10.154.162.141`, interpreter
`/data41/audreymao/a17_vllm/venv/bin/python`, scratch `/data41/audreymao/p18_20261106/`. Code snapshot in
`code/` (the shipped `src` copied from p17). Commands:

- `code/check_grid.py` (brute-force check; prints FAILS 0)
- `code/analyze.py --model llama32_1b --model-dir /data41/audreymao/perf_20261030/bf16model --workers 24 --out out`
- `code/analyze.py --model llama31_8b --model-dir /data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f --layers 0,15,31 --workers 24 --out out`
- `code/summarize_grid.py out llama32_1b llama31_8b` (writes `out/*_grid.csv`)
- `code/distinct_grid.py MODEL_DIR TAG OUT_JSON` (first pass, block 32 and 16; writes `out/distinct_1b.json`)
- `code/free_rows.py MODEL_DIR TAG OUT_JSONL` (free grid, all six block sizes, per-row counts and class sizes;
  `--check` runs the brute-force comparison first)
- `code/aggregate_rows.py OUT_JSONL OUT_JSON` (percentiles, distributions, accumulator and int32 tables)
- `code/tile_grid.py MODEL_DIR TAG OUT_JSONL` (2D tile shapes, `--check` runs the brute-force comparison)
- `code/aggregate_tiles.py OUT_JSONL OUT_JSON` (tile tables)

Raw JSON-lines and CSVs stay on the host. The scripts in this directory are `analyze.py` (p17 plus aligned
grids), `check_grid.py`, `summarize_grid.py`, `distinct_grid.py`, `free_rows.py`, `aggregate_rows.py`,
`tile_grid.py` and `aggregate_tiles.py`.
