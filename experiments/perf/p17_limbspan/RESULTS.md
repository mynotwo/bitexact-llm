# p17: per-block weight limb spans for exact mode

A measurement, not a code change. Exact mode stores every bf16 weight as 3 signed int8 limbs on a grid
shared by one output row, so each bf16 multiply-accumulate costs 9 int8 products (3 weight limbs x 3
activation limbs). That arithmetic is why large-batch decode is 3.7x slower than stock vLLM at 8B. This study
asks how many limbs would be needed if the grid exponent were chosen per block of K instead of per row.

## Headline

A single int8 limb can never hold a bf16 weight. A bf16 significand is 8 bits; a signed int8 limb holds 7.
No block size changes this: the fraction of blocks with span <= 7 is 0 at every block size measured. The
floor of this direction is therefore 2 limbs, 6 products, a 33% arithmetic reduction that is never reached.

Measured mean limbs per weight and the resulting product count, literal rule (smallest L in {1,2,3} whose
residual count is within the block's budget), residual budget 1e-4 x block nonzeros:

| Block along K | row (today) | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1B mean limbs | 3.000 | 2.893 | 2.663 | 2.404 | 2.212 | 2.099 | 2.042 |
| 1B products (x3) | 9.00 | 8.68 | 7.99 | 7.21 | 6.63 | 6.30 | 6.13 |
| 8B mean limbs | 3.000 | 2.916 | 2.683 | 2.404 | 2.206 | not run | not run |
| 8B products (x3) | 9.00 | 8.75 | 8.05 | 7.21 | 6.62 | not run | not run |

1B covers all 16 layers and all 7 linear types; 8B covers all 32 layers and all 7 types. The lm_head of 8B
had not finished when this report was written and is excluded from the 8B row; 1B's lm_head is excluded from
the 1B row as well, so the two rows are comparable.

Block 32 captures most of the available reduction (6.63 of a 6.00 floor) and halving further pays little:
16 gives 6.30 and 8 gives 6.13, while the per-block bookkeeping doubles each time.

## Span distribution (1B, all layers, all 7 types)

`span` is top - bot over the nonzero weights of a block, where top is max(ex + bitlen(mag)) and bot is
min(ex + lowbit(mag)): the number of bits a shared fixed-point grid needs to hold every weight exactly.
Signed base-256 capacities are 7 bits for 1 limb, 15 for 2, 23 for 3.

| Block | blocks | span <= 7 | span <= 15 | span <= 23 |
|---|---:|---:|---:|---:|
| row | 376,832 | 0.000 | 0.000 | 0.8777 |
| 256 | 3,801,088 | 0.000 | 0.107 | 0.9905 |
| 128 | 7,602,176 | 0.000 | 0.337 | 0.9956 |
| 64 | 15,204,352 | 0.000 | 0.596 | 0.9980 |
| 32 | 30,408,704 | 0.000 | 0.788 | 0.9991 |
| 16 | 60,817,408 | 0.000 | 0.901 | 0.9996 |
| 8 | 121,634,816 | 0.000 | 0.958 | 0.9998 |

## No tensor type is the bottleneck

8B mean limbs by type (row / 256 / 128 / 64 / 32):

| Type | row | 256 | 128 | 64 | 32 |
|---|---:|---:|---:|---:|---:|
| q | 3.00 | 2.91 | 2.68 | 2.42 | 2.23 |
| k | 3.00 | 2.91 | 2.70 | 2.43 | 2.23 |
| v | 3.00 | 2.92 | 2.70 | 2.43 | 2.23 |
| o | 3.00 | 2.90 | 2.65 | 2.39 | 2.20 |
| gate | 3.00 | 2.91 | 2.68 | 2.41 | 2.21 |
| up | 3.00 | 2.92 | 2.69 | 2.40 | 2.20 |
| down | 3.00 | 2.93 | 2.69 | 2.40 | 2.20 |

The spread across types is 0.03 limbs at block 32. There is no worst shape to target separately.

## Cross-check against the shipped scheme

Applying the row grid with cap 23, the definition above reproduces the shipped residual density: 5.28e-5 of
nonzero entries for 1B by this study's definition and 5.3e-5 from the shipped `decompose()`, 5.16e-5 for 8B.
The shipped figure is about 5e-5. Every `decompose()` reconstruction compared bit for bit against the bf16
source. The block arrays were also checked against a brute-force Python implementation on a random matrix.

## Residual budget sensitivity

The literal rule gives identical mean limbs at budgets 1e-5, 1e-4 and 1e-3: at these block sizes the
per-block allowance is below one entry, so the rule demands zero residual inside the block and the budget
value does not bind. A pooled rule (one budget per tensor, spent where it buys the most) is infeasible at
1e-5 because the 3-limb residual floor is itself 2.9e-5 (block 32) to 5.3e-5 (row); reaching 1e-5 would need
a 4th limb for outlier blocks, outside this study's {1,2,3} scope and not computed. At 1e-3 the pooled rule
reproduces the literal numbers (2.21 at block 32). The reduction therefore does not depend on budget choice.

## What the reduction costs

Two costs that the limb count alone does not show.

Side data: one grid exponent per block. At 8 bits per exponent that is 8/B bits per weight: 0.03 at block
256, 0.25 at 32, 1.0 at 8. Against 17.7 bits of limbs per weight at block 32 this is 1.4% more storage.
Negligible.

Accumulation: products whose grids differ cannot share an int32 accumulator. Today's kernel keeps one
accumulator per shift class (the 5 diagonals i + j of the 9 products) and combines once per output element.
Per-block grids multiply the shift classes by the number of distinct block exponents, so in the direct
formulation each output element needs K/B shift-and-add steps in int64 on the ordinary ALU:

| Block | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| K = 2048 | 8 | 16 | 32 | 64 | 128 | 256 |
| K = 14336 | 56 | 112 | 224 | 448 | 896 | 1792 |

An order-of-magnitude estimate, not a measurement: at K = 14336 and block 32 the saving is 14336 x (9 -
6.62) = 34,100 int8 tensor-core MACs per output element, worth about 1.7e-11 s at H800's 2000 TOPS int8,
while the added 448 int64 shift-and-adds cost about 3.7e-11 s at an assumed 10-15 Tops effective int64 rate.
On these figures the direct formulation is net negative at large K, and the balance does not improve at
larger blocks because the saving falls as fast as the overhead.

## What this supports and what it does not

Supports: per-block grids cut mean limbs per weight from 3.00 to 2.21 at block 32, products from 9 to 6.6,
on both 1B and 8B, uniformly across tensor types, independent of the residual budget. The floor of the
direction is 2 limbs and 6 products, set by bf16's 8-bit significand against a 7-bit signed limb.

Does not support: any 1-limb result; any throughput claim, since these are storage counts and no kernel was
written or timed; the direct per-block formulation being worth building, which the accumulation estimate
argues against; anything about the activation side, whose 3 limbs come from the 2^-24 residual-stream grid
and would need a trace of real activations to study; the 8B lm_head, which had not finished.

Not yet measured, and the one variant that would avoid the extra accumulation entirely: restricting block
grid exponents to multiples of 8. A whole-limb offset moves a block's contribution to a different diagonal
instead of requiring its own shift class, so the accumulator structure and the single per-output combine
stay as they are today. The cost is up to 7 bits of alignment freedom, which raises the limb count by an
amount these span histograms alone do not determine. That measurement decides whether this direction is
worth a kernel.

## Reproduction

`analyze.py` reads a bf16 HF checkpoint tensor by tensor and emits one JSON line per tensor with the span
histogram and per-budget limb counts for each block size, using the shipped `bf16_parts`, `_bitlen`,
`_lowbit` and `decompose` from `src/bitexact/exact/consts.py`. `summarize.py` aggregates to CSV; counts are
additive across tensors, so aggregates sum counts and recompute ratios, and span quantiles come from summed
integer histograms.

CPU only, no GPU used. Host `audreymao@10.154.162.141`, interpreter
`/data41/audreymao/a17_vllm/venv/bin/python`, scratch `/data41/audreymao/p17_20261106/`: `out/` holds the
1B run with block sizes row through 32 and the 8B run, `out_b8/` the 1B rerun that adds 16 and 8. Raw
JSON-lines output stays on the host.
