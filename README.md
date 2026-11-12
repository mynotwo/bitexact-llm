# BitExact LLM

**Run an unmodified bf16 Llama and get the same logits, bit for bit, on any GPU or CPU, at any batch size
and any KV-cache schedule.**

BitExact replaces every floating-point reduction in Llama inference with integer arithmetic that cannot
overflow, so summation order stops mattering. The weights are the original bf16 values, multiplied
exactly; only activations are carried in fixed point (mean logit difference from HF fp32 about 5e-5,
against 3e-2 for bf16). The cost is speed, so the useful setting is where agreement is worth more than
throughput: low-concurrency serving, RL sampler/trainer agreement, replaying a failure, auditing an output
on different hardware.

Personal research project by Yu (Audrey) Mao.
[Technical article](https://mynotwo.github.io/blog/bitexact-inference/) ·
[CPU checks](https://github.com/mynotwo/bitexact-llm/actions/workflows/cpu.yml) ·
[Reproduction guide](docs/REPRODUCING.md) · [Recorded results](docs/RESULTS.md) ·
[Homepage](https://mynotwo.github.io/)

## Try it in a minute, without model weights

```bash
git clone https://github.com/mynotwo/bitexact-llm.git && cd bitexact-llm
python -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[test]'
python -m bitexact.repro self-check --out arithmetic.json         # 8 integer-arithmetic fixtures
python -m bitexact.repro record --synthetic --out synth.json      # whole model, 64 positions
pytest -q                                                         # includes tests/test_synthetic.py
```

`--synthetic` runs a small 2-layer Llama whose bf16 weights are assembled bit by bit from PCG64 raw words,
through the same constants build, forward pass, KV cache and integer sampler as a real checkpoint.
`tests/test_synthetic.py` pins its logits digests to values recorded on Apple M4 Pro (arm64); CI recomputes
them on Linux x86_64. Send your `synth.json` to anyone and `python -m bitexact.repro compare` tells you
whether their machine produced the same 64 logits rows.

## What is in the package

| Module | Role |
|---|---|
| `bitexact.intops`, `bitexact.sampling` | Integer rules shared by every engine: rounding shifts, floor division, isqrt, limb split/join, first-argmax, integer Gumbel-max keyed by (seed, token index) |
| `bitexact.exact` | **The engine.** Original bf16 weights split into int8 limbs plus a sparse residual (bit-for-bit), int64 fixed-point activations, exact GEMMs on Triton, cuBLAS `_int_mm` or CPU float64 |
| `bitexact.repro` | Identity reports: `self-check`, `record` (also checks incremental vs. batched full forward), `compare` |
| `bitexact.synthetic` | Platform-independent synthetic checkpoint for weight-free whole-model checks |
| `bitexact.vllm_plugin` | The integer engines inside vLLM 0.32.0 (paged KV, prefix cache, CUDA graphs; exact and int8 modes) |
| `bitexact.int8` | A quantized variant with int8 weights. Faster, but it is a different model (logit error about 3x bf16's), not the bf16 model made reproducible |

## Use it with your own checkpoint

Model weights are not included (Llama 3.2 Community License). Build the integer constants once on a CPU;
their SHA-256 identifies the integer model:

```bash
pip install -e ".[triton]"     # CUDA kernels; ".[vllm]" for the plugin
python examples/build_constants.py --model PATH/Llama-3.2-1B --mode exact --out exact_const.pt
python examples/generate.py --const exact_const.pt --tokenizer PATH/Llama-3.2-1B \
    --prompt "The lighthouse keeper wrote" --max-new 64 --temperature 0.7 --seed 1
```

```python
from bitexact import generate, load
model = load("exact_const.pt", "cuda")          # or "cpu"; gemm="cublas", compiled=True are options
tokens, hashes = generate(model, prompt_ids, max_new=64, tn=7, td=10, seed=1)   # T = 7/10
```

Check identity across machines (record on each, then compare; exit code 1 means some row differs):

```bash
python -m bitexact.repro record --const exact_const.pt --device cuda:0 --out gpu.json
python -m bitexact.repro record --const exact_const.pt --device cpu --out cpu.json
python -m bitexact.repro compare gpu.json cpu.json
```

Serving through vLLM (`make_model_dir` writes a model directory vLLM loads with the plugin):

```bash
python -m bitexact.vllm_plugin.make_model_dir --hf PATH/Llama-3.2-1B --const exact_const.pt --out intmodel
python examples/vllm_offline.py --model intmodel --out run1
GPUS="1 2 3" MODEL=intmodel CONST=exact_const.pt bash tests/vllm/run_tests.sh TAG    # invariance suite
```

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
`model.check()` raises `OverflowError` if a checked bound was violated. Exact-mode o_proj / down_proj
outputs are now checked before residual addition; final logits are checked too. The linear epilogue
also flags left-shift bit loss and rejects unsupported positive shifts above 30 for nonzero inputs.
These guards retain the preceding GEMM and residual-accumulation input contracts; they are not a proof
that every arithmetic operation is overflow-free. The recorded model runs in `docs/RESULTS.md` predate these guards;
the added CPU regressions do not re-establish whole-model or CUDA identity for the changed source.

## What is and is not established

- Recorded runs ([docs/RESULTS.md](docs/RESULTS.md)) produced identical int64 logits and sampled tokens for
  Llama-3.2-1B and Llama-3.1-8B across H800, H100, L20, x86 CPU and Apple M4 Pro, batch 1/4/8, incremental
  KV cache, compiled and cuBLAS paths, and inside vLLM under mixed batches, chunked prefill, prefix caching
  and preemption. Those runs predate the current bound guards; a fresh GPU run on the current source is
  still owed.
- Every CPU CI run re-establishes the synthetic-model goldens on Linux x86_64 against arm64.
- Exact mode uses the original bf16 weights for every linear layer, including the tied lm_head. The one
  exception is the input embedding, stored at the residual fixed point (unit 2^-24): 0.019% of
  Llama-3.2-1B's entries change, each by at most 2^-25.
- The result approximates the real-valued model (22-bit activations and attention weights). It is
  reproducible, not identical to fp32.
- Speed: exact mode in vLLM reaches about 0.42x stock vLLM bf16 at batch 1 and 1.19x vLLM's own
  batch-invariant mode there; the gap widens with batch size (see RESULTS.md, "Speed").

## Tests

```bash
pytest -q                                   # CPU: arithmetic oracles, repro CLI, synthetic whole model
BITEXACT_EXACT_CONST=exact_const.pt BITEXACT_INT8_CONST=int8_const.pt pytest -q -m slow   # real constants
```

GPU tests (Triton kernels, cuBLAS path, vLLM kernels) run when CUDA is present and skip otherwise. Unit
tests compare each primitive with Python big-integer and `Fraction` references.

## Limitations

- Two models: Llama-3.2-1B (both modes and the vLLM plugin) and Llama-3.1-8B (exact mode, and int8 mode in the
  vLLM plugin). Shapes
  are read from the constants file; other Llama sizes need new bound checks.
- Contexts up to 4,096 positions in exact mode and 2,048 in int8 mode (the RoPE table length; the vLLM
  plugin rebuilds longer tables at load time).
- GPU results were tested on Hopper (H800, H100) and Ada (L20). V100 (sm70) is not supported because Triton
  3.0 cannot compile the kernels; A100 and H20 are untested. The CPU path was tested on x86 and Apple M4 Pro.
- The constants file of exact mode is about 4.8 GB (3 bytes per weight plus residuals and tables).
- Not a serving system: the standalone engines have no paging or scheduler, and prefill is slow. The vLLM
  plugin provides paging, but only through the offline API (see above).

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
