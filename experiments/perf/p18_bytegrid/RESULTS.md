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

Surprising side result: the number of distinct grid exponents in one output row is small. At block 32 with the
free grid it is 2.6 on average (p90 3, max 7), not K/B. p17's accumulation estimate assumed K/B distinct
exponents per row. This count does not support that assumption on 1B, and it is the more important finding here.

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

## Distinct grid exponents per output row (1B, all 7 types)

For each output row, count distinct grid exponents over its nonempty blocks. Free grid: g = top - 23 per block.
8-aligned and 4-aligned: g = m floor(bot / m). The count is the number of accumulator shift classes a row needs.

| Block | free mean (p90, max) | 4-aligned | 8-aligned |
|---|---|---|---|
| 32 | 2.6 (3, 7) | 3.4 (4, 6) | 2.4 (3, 4) |
| 16 | 2.9 (4, 7) | 3.4 (4, 6) | 2.4 (3, 4) |

Per type, free grid at block 32: q 2.5, k 2.6, v 2.5, o 3.0, gate 2.5, up 2.5, down 3.0. Measured on 1B only; the
8B run did not include this count.

The p17 accumulation estimate (K/B shift-and-add per output element) assumed about K/B distinct exponents per
row. This count gives 2.6 per row at block 32 on 1B, against K/B = 64 at K = 2048. The likely reason (inference,
not checked) is that a block's top tracks the row's weight scale, so few grid values occur. Under that reading the
shift-add overhead in p17 is much smaller than its K/B estimate. This is a count, not a timing; the cost of a
per-exponent accumulator in a real kernel is unmeasured.

## Verdict

Byte-aligned grids do not keep enough of the limb reduction to justify a kernel. At block 32 they give 8.55 products
on 1B and 8.78 on 8B, against 6.63 and 6.69 for the free grid and 9.00 for the row grid, so they recover about 5%
of the 9-product cost. The 4-aligned grid recovers about 9% on 1B (8.21) and 16% on 8B (7.55), and it needs one
extra shift class, which is cheap. Neither reaches the free grid's 6.6, and the 8-aligned grid is not usable at
row or 256 size without a fourth limb.

The stronger finding is the distinct-exponent count: the free grid needs about 2.6 exponents per row, not K/B,
so the p17 argument against the direct formulation is weaker than p17 stated. The count is 1B only, and it was
taken on the free grid, which is the grid the kernel would use. Before writing a kernel, a per-exponent-class
accumulator should be timed on the free grid. That is a separate study, not this one.

## What this supports and what it does not

Supports: 8-aligned grids recover little (8.55 products at block 32 on 1B, 8.78 on 8B); 4-aligned grids recover
more (8.21, 7.55) at one extra shift class; the 1-limb case is impossible for any grid; the free grid's distinct
exponent count per row is about 2.6 at block 32 on 1B.

Does not support: any throughput claim, since these are storage counts and no kernel was written or timed; any
result for the full 8B model (only layers 0, 15, 31 were run); the distinct-exponent count on 8B; any claim about
the activation side, whose 3 limbs come from the residual-stream grid.

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
- `code/distinct_grid.py MODEL_DIR TAG OUT_JSON` (writes `out/distinct_1b.json`)

Raw JSON-lines and CSVs stay on the host. The scripts in this directory are `analyze.py` (p17 plus aligned
grids), `check_grid.py`, `summarize_grid.py` and `distinct_grid.py`.
