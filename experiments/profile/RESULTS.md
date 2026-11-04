# Where the BitExact integer engine spends its time

Profile only: nothing in `src/` was changed. One H100 80GB HBM3 (host 138), torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0. Each number below is labelled as a measurement or an estimate. Estimates are derived from the measured shares and are not measured speed-ups.

## Ranked speed-up opportunities

| rank | opportunity | where | gain | basis | bit-exact across batch, TP, hardware |
|---|---|---|---|---|---|
| 1 | Turn on the engine's existing `compiled=True` (Inductor-fused integer segments) in the scorer | scorer | **4.1x** GSM8K b8 (6,675 -> 27,502 tok/s), 3.6x 4k b4 (3,637 -> 12,920), 3.7x 4k b1 (2,416 -> 8,936) | **measured**; logits sha equal to the plugin's on 28,083 GSM8K rows and 16,384 4k rows, both eager and compiled | yes (measured on batch and sequence; integer ops only) |
| 2 | Fix the single-CTA Inductor reduction in vLLM `compute_logits` (lm_head `Segs.post`, dynamic-shape recompile) | plugin decode B=64 | **~1.75x** decode (8.1 ms of an 18.8 ms step) | estimate from the measured kernel time | yes (an integer max/sum gives the same result in any order) |
| 3 | Scorer attention through the plugin's fused two-sweep kernel (`tri_paged.attend`), which skips keys past each tile's last position | scorer | **~1.15x** GSM8K, **~1.5x** 4k b4, on top of rank 1 | estimate: attention is 30% / 61% of the compiled scorer's wall; kernel measured 1.6-2.3x faster with bit-identical num/den | yes (same integers, measured equal) |
| 4 | CUDA graphs for the scorer (bucketed shapes, no per-chunk `.item()`) | scorer | ~1.1x at b8, up to ~2.4x at b1 4k | estimate: upper bound from GPU-idle share (10% / 8.6% / 58%) | yes |
| 5 | Hand-fused norm+quant, SiLU+mul+quant, GEMM epilogue, plus int32 storage where ranges allow | scorer, decode | ~1.2-1.3x GSM8K scorer, ~1.1x 4k scorer | estimate: compiled elementwise is 38% (GSM) / 18% (4k) of wall; storage bytes halve | yes, if bound checks stay |
| 6 | Integer flash attention with one online-softmax sweep in a spec-defined fixed key-block order | scorer at long context, decode at long context | ~1.2-1.3x decode B64 ctx 2304, ~1.1-1.2x 4k scorer beyond rank 3 | estimate: removes the second QK pass and one K read | yes, but it changes the integer softmax (new spec, new reference numbers) |
| 7 | Avoid padding in the scorer (varlen packing) | scorer | ~1.06x GSM8K, ~1.01x 4k | estimate: 9.7% padded tokens on GSM8K, 0.7% on 4k | yes |
| 8 | One activation limb (nl=1) | scorer | ~1.1x GSM8K, ~1.05x 4k, ~1.0x decode | estimate: GEMMs are 17.5% / 8.5% of wall; decode GEMMs are weight-bandwidth bound | yes, but activation precision drops from 15 to 7 bits per row |

Highest gains: for the scorer (the trainer-side log p, likely part of the 30 s vs 16.5 s RL step), rank 1 is a measured 3.6-4.1x from an existing flag. Ranks 3 and 4 then take 61% and 10-58% of what remains. For plugin decode at batch 64, one 8 ms kernel per step is the largest item.

## Setup

- Scorer: `prof_scorer.py`, the same path as `abc_measure/speed_scorer.py --mode engine`: right-padded batches, prefill in 256-position chunks with S = keys up to the chunk end, then lm_head plus a float64 log-softmax per 256-row response chunk with one `.item()` each. Workloads: the 1,024 GSM8K plugin rollouts of abc_measure (`q1_B.json`, 430,739 tokens, batch 8) and the 16 x 4096-token plugin rollouts (`q2_4096_B.json`, 68,947 tokens, batch 1 / 4 / 8). Throughput comes from unprofiled runs over all batches (median of 3, or of 2 for 4k). Profiles cover the first 4 GSM8K batches or the first 4k batch.
- abc_measure's scorer is **eager** (`IntLlama(c, "cuda")`, default `compiled=False`). Its H800 6,501 tok/s matches 6,675 tok/s here, so the eager runs here reproduce the Q4 number.
- Decode outside vLLM: `prof_decode.py` runs `vllm_plugin.graph.decode_forward` (the path vLLM captures) on a paged int32 KV cache in vLLM's layout, followed by `m.logits` and `sampler.integer_sample`. All B rows have context ctx (512 and 2304). Two modes: eager (uncompiled, category ranges) and compiled (the plugin's settings) with and without a CUDA graph of the forward.
- Decode inside vLLM: `prof_vllm_decode.py`, the `speed_sampler.py` setup (BitExact model, vLLM's FULL_DECODE_ONLY graphs, T=1, ignore_eos, prompts of 178-260 tokens). Measures 512 new tokens, median of 3, then torch.profiler over 128 new tokens.
- Attribution: torch.profiler chrome traces. Each kernel is mapped to the innermost `cat:` range around its launch (`instrument.py`). Eager runs carry fine ranges (RMSNorm, quant_rows, limb split, LUTs). Compiled runs carry only segment-level ranges, because ranges inside a compiled region are dropped. "% of wall" uses the profiled wall time. "GPU idle" is wall minus the union of kernel intervals: launch, Python and host-sync time.
- Bytes: a TorchDispatchMode counts the inputs and outputs of every eager aten op once, with gathers and scatters counted by rows touched. The Triton bmm counts inputs plus output. Paged attention counts K twice and V once at int32. These are logical bytes that assume no cache reuse. They are not DRAM counters: ncu and nsys GPU metrics are blocked on 138 (`RmProfilingAdminOnly: 1`), and nsys fails there because `/tmp` is read-only ("Cannot create CUDA config file"). Compiled runs have no byte count.
- int8 peak: H100 SXM dense int8 tensor core, 1,979 TOPS.

## What the code does (checked against the code and the traces)

- **Linear layers: 2 int8 dots, not 9.** Activations are quantized per row to `nl` = 2 signed base-256 limbs (|q| <= 32,512; plugin `bitexact_nl: 2`, scorer default 2). Weights are 1 int8 limb. The limbs are stacked along rows into one `torch._int_mm` with M = 2 x rows (padded to >= 24, multiple of 8). Trace shapes confirm this, e.g. M = 4096 for 2,048 rows. The int64 residual is never multiplied directly.
- **Attention: 9 int8 dots each for QK and PV.** `bmm.py` (scorer) and `tri_paged.py` (plugin) split q, k, v and p into 3 limbs and run 3 x 3 `tl.dot` with 5 int32 accumulators. Measured ranges support 3 limbs: q and k after RoPE reach 19 bits, v 17 bits, p 21 bits (2 limbs hold 15 bits).
- **Elementwise ops** in the scorer as abc_measure ran it are unfused eager int64 PyTorch ops: about 8 kernels per token at b8 and 50 per token at b1 4k. The plugin and `compiled=True` run them as Inductor-fused Triton kernels.
- **Attention scores:** the scorer materializes int64 q.k scores `[B, 8, 4 x 256, S]` per chunk, and computes all S keys for every query of the chunk (masked afterwards). The plugin does not: its fused two-sweep kernels keep scores in registers.
- **GEMM kernel:** `torch._int_mm` dispatches to `cutlass_80_tensorop_i16832gemm_s8_*`, an sm80 mma.sync kernel, not a Hopper wgmma kernel.

## Scorer: time share (% of profiled wall)

eager = as abc_measure ran it. compiled = `IntLlama(..., compiled=True)`, an existing option. In eager runs the "SiLU x up multiply", "RMSNorm + quantization" and "num/den divide" rows hold only what is left outside the finer ranges. In compiled runs those rows hold the whole fused segment, including its quantization and LUT.

| category (% of wall) | GSM8K b8 eager | GSM8K b8 compiled | 4k b4 eager | 4k b4 compiled | 4k b1 eager | 4k b1 compiled |
|---|---|---|---|---|---|---|
| int GEMM, linear layers (2 limbs = 2 dots) | 3.8 | 15.3 | 1.8 | 6.7 | 0.8 | 3.5 |
| int GEMM, lm_head | 0.5 | 2.2 | 0.5 | 1.8 | 0.2 | 0.7 |
| attention QK (bmm, 9 dots) | 3.9 | 16.5 | 10.3 | 39.0 | 3.7 | 17.8 |
| attention PV (bmm, 9 dots) | 1.2 | 4.7 | 2.5 | 9.0 | 1.0 | 4.3 |
| attention softmax (mask, max, shift, sum; compiled: + exp LUT) | 7.4 | 4.9 | 23.0 | 11.7 | 7.9 | 4.7 |
| exp LUT | 3.8 | (in softmax) | 12.2 | (in softmax) | 4.2 | (in softmax) |
| attention num/den divide (compiled: + quantization) | 0.8 | 3.7 | 0.4 | 1.7 | 0.2 | 0.8 |
| RMSNorm | 3.0 | - | 1.7 | - | 1.7 | - |
| RMSNorm + quantization (compiled segment) | 0.5 | 5.9 | 0.3 | 2.7 | 0.2 | 1.2 |
| quantization (quant_rows, limb split) | 21.9 | (in segments) | 10.8 | (in segments) | 5.3 | (in segments) |
| SiLU LUT | 10.0 | (in SiLU x up) | 4.9 | (in SiLU x up) | 1.4 | (in SiLU x up) |
| SiLU x up multiply (compiled: + SiLU LUT + quantization) | 3.3 | 14.2 | 1.6 | 6.2 | 0.5 | 2.1 |
| GEMM epilogue (limb join, rescale, residual add) | 24.3 | 14.9 | 13.0 | 7.4 | 5.2 | 3.6 |
| RoPE / q,k,v split | 1.6 | 2.1 | 0.8 | 1.0 | 0.6 | 0.9 |
| other elementwise (KV write, flags, embedding) | 0.4 | 1.4 | 0.3 | 0.9 | 0.2 | 1.0 |
| log-softmax float64 of the logits (scorer side) | 1.0 | 4.1 | 0.9 | 3.4 | 0.3 | 1.3 |
| **GPU idle: launch, Python, host sync** | 12.7 | 10.0 | 15.1 | 8.6 | 66.5 | 58.2 |
| scorer tok/s, unprofiled (H100) | 6,675 | 27,502 | 3,637 | 12,920 | 2,416 | 8,936 |

Grouped shares for the compiled scorer, the baseline for the estimates:

| group (% of wall) | GSM8K b8 | 4k b4 | 4k b1 |
|---|---|---|---|
| int GEMMs (linear + lm_head) | 17.5 | 8.5 | 4.2 |
| attention (QK + PV + softmax/exp + divide) | 29.8 | 61.4 | 27.6 |
| RMSNorm + quantization + SiLU + epilogue + RoPE + other | 38.5 | 18.2 | 8.8 |
| log-softmax | 4.1 | 3.4 | 1.3 |
| GPU idle | 10.0 | 8.6 | 58.2 |

Kernel counts per scored token: eager 8.1 (b8), 13.3 (4k b4), 50.3 (4k b1). Compiled: 0.6, 1.0, 3.6.

Padding (measured on the batch layout): GSM8K b8 computes 476,984 token rows for 430,739 real ones (90.3% used). It computes 167.5M query-key pairs for 94.2M causal ones (56.2%: every chunk query sees all S keys, plus padding). 4k b4: 99.3% of tokens and 93.2% of pairs are used.

### int8 GEMMs: achieved vs peak (GSM8K b8, same in eager and compiled)

| M (2 limbs x rows) | N | K | layer | us each | TOPS | % of 1,979 |
|---|---|---|---|---|---|---|
| 4096 | 16384 | 2048 | gate+up | 389 | 706 | 36 |
| 4096 | 2048 | 8192 | down | 145 | 948 | 48 |
| 4096 | 3072 | 2048 | q,k,v | 77 | 673 | 34 |
| 4096 | 2048 | 2048 | o | 52 | 665 | 34 |
| 512 | 128256 | 2048 | lm_head (256 rows) | 390 | 690 | 35 |

All prefill GEMMs reach 33-48% of dense int8 peak with the sm80 CUTLASS kernel. GEMMs are 17.5% of the compiled scorer's wall. A Hopper int8 kernel at about 70% of peak would save about 6-8% of wall (estimate). In decode the GEMMs are bound by weight bandwidth: B=1 pads to M = 24 rows and the lm_head reaches 131 TOPS; B=64 is M = 128 and reaches 524 TOPS.

### Logical bytes per token, eager scorer (MB per real token)

| category | GSM8K b8 | 4k b4 |
|---|---|---|
| GEMM epilogue (limb join, rescale, residual add) | 108.0 | 109.8 |
| quantization (quant_rows, limb split) | 87.9 | 77.7 |
| SiLU LUT + SiLU x up | 66.4 | 58.5 |
| attention softmax + exp LUT (int64 scores) | 61.8 | 350.3 |
| RMSNorm | 8.5 | 7.6 |
| attention bmm QK + PV (inputs + output) | 4.0 | 19.9 |
| int GEMMs (int8 limbs + int8 weights + int32 output) | 5.7 | 6.4 |
| other (RoPE, divide, log-softmax, flags) | 13.7 | 15.4 |
| **total** | **356** | **646** |

Most of the traffic is int64 elementwise passes, not GEMM operands. At 4k context the int64 score tensor and its softmax passes are 54% of the bytes.

### Measured value ranges (max |x| over the GSM8K and 4k runs, bits)

| tensor | bits | fits int32 |
|---|---|---|
| residual stream x | 24-25 | yes |
| int_mm output (int32 accumulator) | 21 | yes |
| joined limb accumulator (post input) | 29 | yes |
| q, k after RoPE / v | 19 / 17 | yes |
| q.k scores (raw) | 37-38 | no |
| p (attention weights) | 21 | yes |
| den = sum p | 28 (GSM8K) / 31 (4k) | at the limit at 4k |
| num = sum p v | 42-46 | no |
| gate/up output | 22-23 | yes |
| logits | 22 | yes |
| largest rounding-shift input (acc x m, t x mant) | 63 | no (int64 products, register-only) |

## Plugin decode

### In vLLM (FULL_DECODE_ONLY graphs), measured

| batch | tok/s (512 new) | ms per step | GPU busy in profile | kernels per step |
|---|---|---|---|---|
| 1 | 234 | 4.27 | 40% | 595 |
| 64 | 3,535 | 18.1 | 72% | 898 |

Kernel classes over the profiled 128-token generate (prefill included, about 12% of kernel time at B=64):

| kernel class (% of GPU kernel time) | B=1 | B=64 |
|---|---|---|
| Inductor fused elementwise | 39.5 | **81.4** |
| int GEMM (cuBLASLt `_int_mm`) | 34.9 | 6.4 |
| eager elementwise | 15.7 | 3.7 |
| attention (fused paged, two sweeps) | 8.3 | 7.7 |
| other | 1.6 | 0.8 |

**One kernel dominates B=64.** In the decode phase (127 steps after the first graph replay), one Inductor reduction kernel takes 8.1 ms per step: 1,028.8 of 1,716.5 ms of decode kernel time, against a median step of 18.8 ms. It is `triton_red_fused_..._rsub_slice_sub_unsqueeze_where_zeros_like_0`, launched with grid (1,1,1) and block 512, right after the lm_head GEMM. Its enclosing region is "Torch-Compiled Region: 1/2", the third compilation of the `Segs.post` lambda (`TORCH_LOGS=recompiles`: recompiled on a size change of `c`, so dynamic shapes). That makes it compute_logits' lm_head epilogue (limb join, rescale, overflow-flag amax over 64 x 128,256), run by a single CTA. The same segment takes 0.09 ms in `prof_decode.py` and 0.29-0.32 ms in `dyn_logits.py`, even after forcing dynamic row and width recompiles. Why vLLM's compile produces a single-CTA reduction was not found (no root-cause evidence). `split_reductions` is True after importing vLLM.

### Outside vLLM: one decode step, measured

| B, ctx | eager step (ms) | compiled, no graph | compiled forward, graph | compiled step, graph (+ logits + sampler) |
|---|---|---|---|---|
| 1, 512 | 107 | 11.0 | 2.40 | 2.89 |
| 1, 2304 | 106 | 11.1 | 2.54 | 3.03 |
| 64, 512 | 116 | 13.7 | 5.21 | 6.94 |
| 64, 2304 | 115 | 14.1 | 10.89 | 12.64 |

Compiled + graph, kernel time per step (ms):

| class | B1 c512 | B1 c2304 | B64 c512 | B64 c2304 |
|---|---|---|---|---|
| attention, fused paged | 0.21 | 0.34 | 1.90 | **7.59** |
| Inductor fused elementwise (layers) | 0.90 | 0.91 | 1.78 | 1.79 |
| int GEMM, layers | 0.76 | 0.76 | 0.85 | 0.85 |
| eager elementwise (KV write, index) | 0.36 | 0.36 | 0.48 | 0.48 |
| lm_head GEMM + epilogue | 0.12 | 0.12 | 0.22 | 0.22 |
| sampler (Gumbel-max) | 0.04 | 0.04 | 0.45 | 0.45 |
| GPU busy / wall (profiled) | 2.38 / 4.29 | 2.52 / 4.35 | 5.68 / 8.30 | 11.37 / 14.07 |

Eager decode (B=64, ctx 2304) is 84% GPU idle (12,556 kernels per step). Its kernel time splits into quantization 23%, attention 22%, sampler 19%, epilogue 14% and RMSNorm 11%. This mode only shows where unfused time would go; the plugin does not run it.

Logical bytes per token, eager decode: B=64 ctx 2304 826 MB, of which the sampler is 311 MB (int64 [64, 128,256] Gumbel rows), paged attention 229 MB (int32 K twice and V once over 2,304 keys x 16 layers) and the GEMM epilogue 111 MB. B=1 ctx 512 1,940 MB, of which 1,290 MB are int8 weights. At B=64 ctx 2304 the attention kernels move about 14.7 GB per step in 7.6 ms, about 1.9 TB/s or 58% of HBM3 bandwidth, so they are not purely bandwidth bound: QK runs 9 dots in each of the two sweeps.

## Candidates: estimates and exactness

All gains below are estimates computed from the measured shares unless marked measured. "Exact" means the logits stay bit-identical across batch composition, TP and GPU type. Any rule built from integer adds, shifts, maxima and integer sums keeps that, because the result does not depend on order or tiling. The engine's invariance argument (`int8/model.py`, `paged.py`) covers every candidate except the one that redefines softmax.

1. **Fused Triton kernels for norm+quant and SiLU+mul+quant.** Inductor already does this fusion. The existing `compiled=True` gives a measured 4.1x / 3.6x / 3.7x over the eager scorer, and logits sha equal the plugin's on 44,467 rows. Beyond Inductor, compiled elementwise is still 38.5% of GSM8K wall and 18.2% of 4k wall. Hand fusion of quantization into the GEMM prologue and of join/rescale/residual into the GEMM epilogue would remove the int64 [M, N] round trips (e.g. gate/up written as int64 [M, 16384]). Estimate: about half of that share, 1.2-1.25x GSM8K and 1.1x 4k. Exact: yes.
2. **int32 instead of int64 where ranges allow.** Measured ranges allow int32 storage for the residual (25 bits), q/k/v (19 bits), p (21), gate/up (23), logits (22) and joined accumulators (29). They do not allow it for raw scores (38 bits), num (46 bits), den (31 bits at 4k, at the int32 limit) or the products inside rounding shifts (63 bits, which can stay in registers). The gain overlaps with item 1, since both cut bytes of the same passes. Estimate: 1.1-1.2x for the scorer, and about 1.3x combined with item 1. Exact: yes, if the existing on-device bound flags are kept (a value that overflowed int32 would otherwise be silent). den needs int64 beyond about 4k context.
3. **Fewer activation limbs.** The engine already uses 2 limbs, not 3. Going to 1 limb halves GEMM M. GEMMs are 17.5% (GSM8K) / 8.5% (4k) of compiled scorer wall, so the estimate is 1.10x / 1.04x. Decode gains about nothing: B=1 is padded to 24 rows and both B=1 and B=64 are weight-bandwidth bound. Precision cost: per-row activation quantization drops from 32,512 levels (15 bits) to 127 levels (7 bits), i.e. W8A8 instead of W8A16. abc_measure's B-vs-fp32 residual (mean |d| 0.0012) includes activation quantization at 2 limbs; 1 limb would raise it, by an unmeasured amount. The integer multipliers `fm` assume up to 3 limbs and would stay valid. Exact: yes, but it is a different integer model with new reference logits.
4. **Fused integer flash attention with a fixed, spec-defined block order.** Two steps.
   - (a) Use the plugin's existing fused two-sweep kernel in the scorer. Measured on the same integer q/K/V, num and den are bit-identical to the scorer's unfused path. Kernel speed-up is 1.59x / 2.05x / 2.29x for chunk ends 256 / 1,280 / 4,096 at B=4 and 1.76x / 1.94x at B=8 (S = 256 / 512). It also skips most of the 44% of query-key pairs that are causally masked on GSM8K: keys are bounded by each 64-query tile's last position. Estimate: 1.15x GSM8K, 1.5x 4k b4. Exact: yes, with no spec change.
   - (b) A single online-softmax sweep. As `tri_paged.py` notes, the engine's softmax quantizes each weight against the final row maximum, and the integer exp table does not satisfy exp(a)·exp(b) = exp(a+b), so rescaling partial sums cannot reproduce today's integers. A spec that defines key blocks at fixed absolute positions (e.g. 64 keys), a per-block maximum and an integer rescale rule (table lookup of the max difference, fixed rounding) would be batch-, TP- and hardware-invariant by the same argument as vLLM's batch-invariant kernels. Block boundaries would come from absolute positions, not from batch layout. It changes the model's softmax, so the plugin and the scorer must switch together and new reference numbers are needed. It saves the second QK pass (9 dots) and one K read. Estimate: about 35-40% off attention time. Decode B64 ctx 2304: attention is 60% of the graphed step, so about 1.25x. 4k scorer: about 1.1-1.2x beyond (a). KV-cache int32 (2x bf16 bytes) is not changed by this.
5. **CUDA graphs for the scorer.** GPU idle in the compiled scorer is 10.0% (GSM8K b8), 8.6% (4k b4) and 58.2% (4k b1). Upper-bound estimates: 1.11x, 1.09x and 2.4x. Graphs need fixed shapes (bucket by length, like `graph.py`'s row buckets) and no host syncs inside: drop the per-chunk `.item()` of the log-prob and keep `S` fixed, as `forward(S=...)` already allows. The decode path shows the size of this effect: compiled forward 11.7 ms without a graph vs 5.2 ms with one (B=64, ctx 512). Exact: yes (same kernels and integers; padding rows are row-local).
6. **Avoiding padding in the scorer.** 9.7% of computed token rows are padding on GSM8K b8 and 0.7% on 4k. Removing them saves that share of the non-attention time (60% of wall): estimate 1.06x GSM8K, 1.01x 4k. Most of the attention waste on GSM8K is causal masking (56% of pairs used), which item 4a removes. Exact: yes. The engine already states that logits do not depend on padding or batch neighbours.

Not on the list but measured:
- The vLLM B=64 `compute_logits` kernel (8.1 ms per step, single CTA). Estimated decode gain 1.75x at B=64. Exact: yes.
- B=1 decode is host bound: 4.27 ms per step in vLLM vs 2.89 ms for the graphed forward + logits + sampler alone, and 60% GPU idle in the profile. The integer sampler does host round trips (`.tolist()`, per-row seeds to `gpu_pcg.raw_rows`). Estimate: up to about 1.5x at B=1 (4.27 / 2.89) if the host gaps shrink to the graph-only level.
- GEMMs run an sm80 CUTLASS int8 kernel at 33-48% of H100 peak. Estimate: 1.06-1.08x for the scorer, since GEMMs are 17.5% of wall.

Rough compound for the GSM8K scorer (estimate, assuming independence): rank 1 (measured 4.1x) x 1.15 (4a) x 1.1 (graphs) x 1.25 (fusion/int32) x 1.06 (padding) gives about 6.9x over today's eager scorer, about 46k tok/s on H100. That would match vLLM batch-invariant's 39-45k and remain below HF bf16's 122k.

## Caveats

- One GPU type (H100); abc_measure's Q4 was on H800. Eager scorer throughput agrees (6,675 vs 6,501 tok/s).
- Profiled wall times include profiler overhead. It is large for eager decode (217 ms profiled vs 115 ms unprofiled per step) and small for the compiled scorer (0.51 s profiled vs 0.50 s implied by unprofiled throughput for the same 4 batches). Shares use the profiled run and throughputs use unprofiled runs.
- Decode outside vLLM uses random int32 K/V and equal context for all rows. Attention cost depends on lengths, not values. The vLLM runs use real prompts with contexts up to about 770 tokens (prompt + 512).
- Bytes are logical (one read of each input and one write of each output per eager op), not DRAM counters. Hardware counters are not available on 138.
- The compiled-scorer exactness check covers 128 GSM8K sequences (28,083 rows) and 4 x 4k sequences (16,384 rows) against the plugin's recorded logits hashes. Eager matched on the same rows.
- Estimates assume the measured shares stay as they are after each change. Gains from different items overlap (e.g. fusion and int32 cut the same bytes; graphs matter less once kernels are fewer).

## Files

- `instrument.py` category ranges, trace attribution, logical byte counter
- `prof_scorer.py`, `prof_decode.py`, `prof_vllm_decode.py`: profiled runs; `run138.sh`, `env138.sh`: the 138 runner
- `check_compiled.py` (compiled vs eager scorer against plugin hashes), `bench_attn.py` (unfused vs fused paged attention, equality and time), `range_probe.py` (value ranges), `dyn_logits.py` (logits epilogue after dynamic recompiles)
- `summarize.py` builds `results/tables.md` (all per-run tables: categories, kernel classes, GEMM shapes, bytes) from `results/*.json`. Raw traces stay on 138 (deleted after attribution, up to 300 MB each).
