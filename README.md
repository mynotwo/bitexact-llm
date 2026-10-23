# BitExact LLM

Bit-reproducible Llama inference with integer arithmetic. The same prompt, seed and model give the same
logits, bit for bit, on an H800, an H100, an x86 CPU and an Apple M4 Pro CPU, at batch size 1 or 8, with or
without the KV cache. The int8 mode also keeps every bit inside a busy vLLM server.

Personal research project by Yu (Audrey) Mao. Homepage: https://mynotwo.github.io/

## What it is

`bitexact` runs Llama-3.2-1B in two integer modes:

- **exact** (`bitexact.exact`): the original bf16 weights, unchanged, multiplied exactly with int64
  fixed-point activations. Its logits differ from HF fp32 by 5e-5 on average (bf16 differs by 3e-2).
- **int8** (`bitexact.int8`): int8 weights, a quantized integer model. It is faster than exact at batch 1
  and in prefill; its logits differ from fp32 by about 9e-2, about 3x bf16's error.

Both share the integer rules, the attention kernel and the sampler. An optional vLLM plugin
(`bitexact.vllm_plugin`) serves the int8 mode through vLLM's scheduler, paged KV cache and prefix cache.
For comparison, `bitexact.det_float` is a float bf16 engine made batch-invariant with fixed Triton tiles.

## Why integers make results order-independent

Floating-point addition is not associative, so a sum depends on the order of its terms. GPU kernels pick
that order by tile size, split-K, batch size and hardware, which is why a token's logits change when the
batch changes or the model moves to another GPU. Batch-invariant kernels fix the order for one kernel
build; they are not designed to make different GPUs or libraries agree.

Integer addition is associative. If every intermediate value is an integer that cannot overflow, any
summation order gives the same result, on any device. `bitexact` keeps every tensor on the forward and
sampling path as an integer with a fixed number of fractional bits and states each rounding as an integer
rule (round half up, floor division, shifts):

- **Matrix products.** Activations are quantized per row with a power-of-two scale (|q| <= 2^22) and split
  into three signed int8 limbs. Each bf16 weight row is split into a dense part (three int8 limbs on a
  per-row grid) plus a sparse residual for the few low bits below that grid, and the split reproduces every
  weight bit for bit. The 9 limb products run on int8 tensor cores with int32 accumulators that cannot
  overflow (|acc| <= 3 * K * 2^14 < 2^31) and are joined in int64. A linear output is the exact product
  `q @ W_bf16`, rounded once. The same integers come from a Triton kernel, from cuBLAS `torch._int_mm`, or
  from a float64 GEMM on the CPU where every partial sum stays below 2^53.
- **Non-linear functions.** RMSNorm uses an integer sum of squares, an exact integer square root and exact
  long division. RoPE uses 30-bit cos/sin tables. Softmax and SiLU use a 2^-f table with linear
  interpolation and integer division. All tables are computed once in float64 and stored in the constants
  file, so no transcendental function runs at inference time.
- **Row-local rules.** Quantization scales, norms and softmax maxima are per row (per token and head), and
  masked keys get weight 0 before any reduction. A token's result therefore does not depend on its batch
  neighbours, on padding, or on whether its keys came from the cache.
- **Sampling.** Greedy picks the first maximum. With T > 0, Gumbel-max adds integer Gumbel noise derived
  from numpy's PCG64 raw words, keyed by (seed, generated-token index), using an integer ln table. The noise
  for a given key is the same on every device and in every batch.

This describes exact mode. int8 mode follows the same scheme with int8 weights (one limb), activations in
two limbs, and direct exp / SiLU tables.

Bounds (|x| < 2^40, KV values within 3 limbs, products < 2^62, ...) are checked on the device, and
`model.check()` raises `OverflowError` if one was violated. Two gaps remain: the o_proj / down_proj outputs
are checked only after they are added to the residual stream, and the final logits are not checked. No
check fired in any of the runs below.

## Guarantees and their scope

- Given the same constants file (identified by its SHA-256), exact mode gives identical int64 logits and
  identical sampled tokens across the devices, software versions and execution modes in the table below.
  This was tested on those configurations only, not proven for all hardware.
- exact mode uses the original bf16 weights for every linear layer, including the tied lm_head (checked
  bit for bit over all 1,235,746,816 elements). **The one exception is the input embedding**: it is stored
  at the residual stream's fixed point (int32, unit 2^-24), so each entry is rounded once to 2^-24.
  50,498 of 262,668,288 entries (0.019%) change, each by at most 2^-25 (about 3e-8).
- The result is still an approximation of the real-valued model (22-bit activations and attention
  weights). It is reproducible, not identical to fp32.

## Install

```
pip install -e .              # CPU path (torch, numpy, safetensors, transformers)
pip install -e ".[triton]"    # CUDA: Triton GEMM and attention kernels
pip install -e ".[vllm]"      # vLLM plugin (tested with vLLM 0.32.0)
```

Model weights are not included. Download Llama-3.2-1B in Hugging Face format (one `model.safetensors`);
the weights are under Meta's Llama 3.2 Community License.

## Usage

Build the integer constants once (CPU, float64). The constants hold every weight and table as integers,
and their SHA-256 identifies the integer model:

```
python examples/build_constants.py --model PATH/Llama-3.2-1B --mode exact --out exact_const.pt
python examples/build_constants.py --model PATH/Llama-3.2-1B --mode int8 --out int8_const.pt
```

Generate:

```
python examples/generate.py --const exact_const.pt --tokenizer PATH/Llama-3.2-1B \
    --prompt "The lighthouse keeper wrote" --max-new 64 --temperature 0.7 --seed 1
```

From Python:

```python
import torch
from bitexact import generate, load

model = load("exact_const.pt", "cuda")          # or "cpu"; gemm="cublas", compiled=True are options
tokens, hashes = generate(model, prompt_ids, max_new=64, tn=7, td=10, seed=1)   # T = 7/10
```

Check bit identity across devices (record on each machine, then compare):

```
python examples/check_bit_identity.py record --const exact_const.pt --device cuda:0 --out h800.json
python examples/check_bit_identity.py record --const exact_const.pt --device cpu --out cpu.json
python examples/check_bit_identity.py compare h800.json cpu.json
```

vLLM (int8 mode):

```
python -m bitexact.vllm_plugin.make_model_dir --hf PATH/Llama-3.2-1B --const int8_const.pt --out intmodel
python examples/vllm_offline.py --model intmodel --out run1
```

## Tests

```
pytest -q                                    # CPU unit tests; GPU tests run when CUDA is present
BITEXACT_EXACT_CONST=exact_const.pt BITEXACT_INT8_CONST=int8_const.pt pytest -q -m slow
```

The unit tests compare each primitive with Python big-integer and `Fraction` references (division,
rounding shifts, two-word combine, quantization, the linear layer with residual against the exact rational
sum over the bf16 weights, all three GEMM paths). The `slow` tests check that one engine gives the same
logits alone, in a padded batch, through the KV cache, with compiled stages, with cuBLAS and on the CPU.
Without constants files they can build a 2-layer model from `BITEXACT_MODEL=PATH/Llama-3.2-1B`.

## Verified results (exact mode)

### Bit identity

Exact mode, Llama-3.2-1B, positions are generated tokens; each position compares the SHA-256 of the full
int64 logits row and the token sampled with integer Gumbel-max at T = 0.7.

Main run: all 3,696 generated positions of 16 samples; reference H800 (card 4), batch 1, one full forward.

| Setting | Device / software | Positions | Logits differ | Tokens differ |
|---|---|---|---|---|
| batch 1, full forward | H800, torch 2.4.1, Triton 3.0 | 3,696 | 0 | 0 |
| batch 8 (right padding), full forward | same card | 3,696 | 0 | 0 |
| prefill + token-by-token KV cache | same card | 3,696 | 0 | 0 |
| same, torch.compile fused stages | same card | 3,696 | 0 | 0 |
| batch 8, cuBLAS `_int_mm` path | same card | 3,696 | 0 | 0 |
| the 5 settings above | another H800 | 3,696 x 5 | 0 | 0 |
| batch 1, batch 8, incremental, incremental + compile | H100, torch 2.6.0, Triton 3.2 | 3,696 x 4 | 0 | 0 |
| batch 1, full forward | x86 CPU (Xeon 8457C, float64 GEMM) | 3,696 | 0 | 0 |
| batch 8, after changing GEMM tiles and the division implementation | H800 | 3,696 x 2 | 0 | 0 |
| batch 1, full forward, 8 threads and 1 thread | Apple M4 Pro (arm64, Apple Accelerate BLAS), torch 2.14.1, CPU path | 3,696 x 2 | 0 | 0 |

The Apple M4 Pro runs match both the H800 and the x86 CPU references. Their constants were built on x86
and only loaded on ARM, so table generation on ARM was not tested.

An independent re-check on 4 new prompts (narrative, code, history, recipe; 13-17 tokens each) generated
128 tokens per prompt through the KV cache with T = 0.7 sampling, so any differing bit would also change the
continuation. Each run also recomputed the same positions in one right-padded batch of 4. Reference: H800,
incremental.

| Run | Environment | Incremental: positions / logits differ / tokens differ | Batch of 4 |
|---|---|---|---|
| H800 Triton | torch 2.4.1, Triton 3.0 | 512 / 0 / 0 | 512 / 0 / 0 |
| H800 Triton, another card | same | 512 / 0 / 0 | 512 / 0 / 0 |
| H800 torch.compile | same | 512 / 0 / 0 | 512 / 0 / 0 |
| H800 cuBLAS `_int_mm` | same | 512 / 0 / 0 | 512 / 0 / 0 |
| H100 Triton | torch 2.6.0, Triton 3.2 | 512 / 0 / 0 | 512 / 0 / 0 |
| H100 torch.compile | same | 512 / 0 / 0 | 512 / 0 / 0 |
| x86 CPU (float64 limb GEMM) | x86_64 | 512 / 0 / 0 | 512 / 0 / 0 |

Constants files built from the same safetensors on two machines (torch 2.4 and torch 2.6) have the same
SHA-256.

### Accuracy

Teacher forcing on generated positions: 100 texts generated by an 8B model at each of three temperatures
(T0, T0.7, T1.0; 95,444 positions), plus 2,044 positions of enwik8 text (4 segments of 512 tokens).
Reference: HF fp32 with TF32 off. KL and cross-entropy in bits/token.

| Data | Model | Mean \|Δlogit\| | Max \|Δlogit\| | Top-1 agree | KL(fp32 ‖ ·) | Cross-entropy |
|---|---|---|---|---|---|---|
| T0 | exact | 5.4e-5 | 0.0018 | 100% | 3.7e-10 | 0.4869 |
| T0 | HF bf16 | 3.0e-2 | 1.05 | 99.45% | 2.6e-4 | 0.4876 |
| T0 | int8 | 9.1e-2 | 2.30 | 99.33% | 9.1e-4 | 0.4882 |
| T0.7 | exact | 5.2e-5 | 0.0017 | 100% | 7.0e-10 | 1.3473 |
| T0.7 | HF bf16 | 2.9e-2 | 0.74 | 98.57% | 5.4e-4 | 1.3482 |
| T0.7 | int8 | 8.8e-2 | 2.54 | 98.48% | 1.8e-3 | 1.3498 |
| T1.0 | exact | 4.5e-5 | 0.0023 | 100% | 1.0e-9 | 3.8581 |
| T1.0 | HF bf16 | 2.6e-2 | 1.20 | 97.52% | 7.8e-4 | 3.8585 |
| T1.0 | int8 | 7.6e-2 | 2.39 | 97.59% | 2.4e-3 | 3.8602 |
| text | exact | 5.9e-5 | 0.0020 | 100% | 1.6e-9 | 3.2194 |
| text | HF bf16 | 2.9e-2 | 0.87 | 98.39% | 7.8e-4 | 3.2190 |
| text | int8 | 8.7e-2 | 2.05 | 97.75% | 3.2e-3 | 3.2239 |

HF fp32's own cross-entropy is 0.4869 / 1.3473 / 3.8581 / 3.2194; exact matches it to < 1e-4 on all four.
Against HF fp64, exact's mean difference is 4.7e-5 to 5.9e-5 (fp32: 3.7e-6 to 5.1e-6; bf16: 2.7e-2 to
2.9e-2), with 100% top-1 agreement. The error is about 10x fp32's and about 1/500 of bf16's, mostly from
the 22-bit activations and attention weights.

An independent check on 2 new texts (358 positions, full vocabulary): exact vs fp32 5.0e-5 mean, 7.9e-4 max,
100% top-1; bf16 vs fp32 2.9e-2 mean, 0.42 max, 97.5% top-1.

### Speed

The value of this project is reproducibility, not throughput. Stock vLLM bf16 (CUDA graphs) decodes 2.5 to
6x faster than the exact engine at batch 8 and above, and prefills about 20x faster. The exact engine is
about as fast as vLLM's batch-invariant mode at batch 8 (1.18x) and slower at batch 1 (0.64x) and at large
batch (0.18x at batch 100).

One H800; 128 input tokens, 128 output tokens; prefill is 8 x 512 tokens. exact and int8 run each decode
step as one CUDA graph with fixed length S = P + N (no paging), median of 3. vLLM runs are vLLM's default
torch.compile + CUDA graphs; BI is `VLLM_BATCH_INVARIANT=1`; its decode-only rate is estimated as
`B * 127 / (t_out128 - t_out1)`, and end-to-end throughput including prefill is in parentheses. Other jobs
were running on the machine (load average 30 to 99) and each vLLM configuration was run once, so absolute
numbers are lower than on an idle machine; comparisons within the table are more reliable.

| Batch | vLLM bf16 decode | vLLM bf16 BI decode | exact decode | int8 decode |
|---|---|---|---|---|
| 1 | 638 (e2e 602) | 209 (e2e 205) | 133 | 328 |
| 8 | 5,157 (e2e 4,483) | 1,746 (e2e 1,682) | 2,062 | 1,642 |
| 32 | 16,724 (e2e 15,237) | invalid run (e2e 7,155) | 4,265 | 2,907 |
| 100 | 28,733 (e2e 26,685) | 26,403 (e2e 24,744) | 4,745 | 3,495 |

8 x 512 prefill, prompt tokens/s: vLLM bf16 186,182; vLLM BI 124,121; exact 9,431; int8 20,579.

Earlier measurements against HF transformers bf16 *eager* (DynamicCache, no CUDA graphs, no torch.compile)
on an idle H800 gave exact 253 / 1,947 / 6,402 tok/s decode at batch 1 / 8 / 100 versus HF 123 / 953 /
10,671. The 2x lead at small batch only holds against that eager baseline, whose per-step time is dominated
by kernel launches.

### vLLM plugin (int8 mode)

vLLM 0.32.0. 12 target prompts (lengths 7 to 700, 6 sharing a 320-token prefix; T = 0 / 0.7 / 1.0; 48
tokens each, 576 positions) were run alone and inside busy mixed batches with 40 filler requests arriving
at random engine steps: chunked prefill with budgets of 64 / 256 / 512 tokens per step, prefix cache on and
off, and a KV cache of 160 blocks that forced 5 preemptions and recomputation. Every run, with and without
the plugin's CUDA graphs and torch.compile, matched the standalone int8 engine with 0 differing token
sequences and 0 differing logits hashes. The same 8 runs on H800 matched the H100 reference with 0
differences.

Throughput on one H100 (128 input + 128 output tokens per request, including prefill):

| Config (output tok/s) | B=1 | B=32 | B=128 |
|---|---|---|---|
| vLLM bf16, CUDA graphs | 659 | 15,113 | 34,733 |
| vLLM bf16 batch-invariant, CUDA graphs | 212 | 7,356 | 19,104 |
| plugin, greedy | 215 | 2,740 | 5,044 |
| plugin, T = 1 | 138 | 1,350 | 1,370 |

At batch 32 to 128 the plugin runs at 15-18% of stock vLLM bf16 (greedy). With T > 0 each step generates
128,256 PCG64 words per request for the noise.

Plugin limitations:

- The published invariance and throughput results use `enforce_eager=True` (the plugin then captures its
  own CUDA graphs for pure decode steps). Running under vLLM's own CUDA graphs is supported in the code but
  not covered by those results. Single GPU only (TP = 1, PP = 1); no speculative decoding.
- Sampling uses only temperature and seed. top-k, top-p, min-p, repetition penalties and logit bias are
  ignored. Requests without a seed use seed 0, so two such requests with the same temperature get the same
  noise.
- logprobs returned by vLLM are computed from the integer logits converted to float32 and are approximate;
  the reproducible outputs are the tokens and the integer logits hashes.
- Tested with the offline `LLM` / `LLMEngine` API only, not `vllm serve`.
- Cross-GPU tests covered H100 and H800 only (both sm90).
- `BITEXACT_COMPILE_LAYER=1` (one torch.compile region per decoder layer) is experimental and off by
  default: replayed inside the plugin's CUDA graphs it changed logits on vLLM 0.32 / torch 2.13.

## Limitations

- One model: Llama-3.2-1B (hidden size 2048, 32 query / 8 KV heads, head dim 64). Shapes are constants in
  `bitexact/llama.py`; other Llama sizes need new bound checks.
- Contexts up to 4,096 positions in exact mode and 2,048 in int8 mode (the RoPE table length; the vLLM
  plugin rebuilds longer tables at load time).
- GPU results were tested on Hopper only (H800, H100). Other GPU architectures are untested; the CPU path
  was tested on x86 and Apple M4 Pro.
- The constants file of exact mode is about 4.8 GB (3 bytes per weight plus residuals and tables).
- Not a serving system: the standalone engines have no paging or scheduler, and prefill is slow.
- `bitexact.det_float` is a float engine: its results are fixed only for one GPU architecture and Triton
  version, unlike the integer modes.

## Related work

- Thinking Machines, *Defeating Nondeterminism in LLM Inference* (2025): batch-invariant RMSNorm, matmul and
  attention kernels; they do not claim agreement across GPUs or PyTorch versions.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- SGLang deterministic mode (LMSYS, 2025): fixed split sizes in the attention backends, 34% average
  slowdown, tested on H100/H200 without a cross-GPU claim.
  https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
- ReproFuse (2026): fixed-configuration GEMMs that make linear layers agree across Ampere/Ada/Hopper;
  attention, RoPE, norm and sampling are outside the guarantee, and full-model logits agree on 9.6% of
  tokens against H100. https://arxiv.org/abs/2609.25624
- Gensyn RepOps / REE: fixed floating-point operation order for cross-hardware bitwise reproducibility
  (Verde, https://arxiv.org/abs/2502.19405). The REE documentation claims bitwise identity on any supported
  hardware but lists neither the tested GPUs nor the overhead. https://docs.gensyn.ai/tech ,
  https://github.com/gensyn-ai/ree

Integer-only transformer inference itself is not new (I-BERT https://arxiv.org/abs/2101.01321 , I-LLM
https://arxiv.org/abs/2405.17849); those works target efficiency rather than reproducibility.

## License

Apache License 2.0 (see `LICENSE`). Model weights are not included and are licensed separately by Meta
under the Llama 3.2 Community License.
