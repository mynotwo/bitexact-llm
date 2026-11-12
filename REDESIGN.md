# Redesign notes

## What the project is for

Bit-reproducible inference for **unmodified bf16 Llama models**: the same logits, bit for bit, on any GPU
or CPU, at any batch size and KV-cache schedule. Exact mode is the product. It keeps the original weights,
so a user does not trade the model for reproducibility. The int8 mode is a different (quantized) model
and stays as a secondary, faster option. Speed was measured and is not where the project wins (about
0.42x stock vLLM at batch 1, falling further behind at larger batches). Its use cases are low-concurrency
serving, RL sampler/trainer agreement, failure replay and cross-hardware audit.

## What was wrong with the old layout

- Nobody could check the central claim without Meta weights, a 4.8 GB constants build and a GPU. The only
  weight-free check covered 8 arithmetic fixtures, not the model.
- The README (628 lines) mixed the method with session-by-session performance logs, so the purpose was hard
  to find.
- `det_float` (a float engine that is reproducible on one GPU type only) sat in the package as if it were
  a peer of the integer engines.

## Changes

- **`bitexact.synthetic`** (new): a 2-layer bf16 Llama (hidden 512, GQA 8/2, vocabulary 512) whose weights
  are assembled bit by bit from PCG64 raw words. Its bytes are the same on every platform, and it
  exercises the sparse-residual weight path. It runs the full exact engine (constants build, forward, KV
  cache, batching, integer Gumbel sampler) in seconds on a laptop.
- **`repro record --synthetic`**: a v1 identity report from that checkpoint, comparable across machines
  with `repro compare`. The sampler and batch verifier now handle vocabularies smaller than Llama's
  (`raw_rows(keys, n)` takes the first n words of the same PCG64 stream, so Llama results are unchanged).
- **`tests/test_synthetic.py`**: pins weight, constants and logits-row SHA-256s recorded on Apple M4 Pro
  (arm64). CI on Linux x86_64 recomputes them, so every CPU CI run checks whole-model bit identity across
  architectures. CI also uploads the synthetic report.
- **Removed** `bitexact/det_float.py` (outside the integer guarantee) and `examples/check_bit_identity.py`
  (a shim around `python -m bitexact.repro`).
- **README** rewritten around the purpose (183 lines): a weight-free quick start, a module map, use with a
  real checkpoint, the method, and what is and is not established. The full recorded tables moved verbatim
  to `docs/RESULTS.md`.
- **Kept**: `exact`, `int8`, `vllm_plugin`, the GPU kernels and tests, and `experiments/`. These hold the
  recorded evidence and the serving path, and they need CUDA, which this redesign could not run. None of
  their code changed.

## How to run

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[test]'
pytest -q                                          # 74 passed, 59 skipped (CUDA / real constants)
python -m bitexact.repro record --synthetic --out synth.json
```

## What I would build next

1. Re-run the GPU identity suite (H800/H100/L20, standalone and vLLM) on the current source and record the
   synthetic report on each GPU too. That turns the synthetic goldens into a GPU check as well.
2. Make the synthetic model larger or rescale it toward real activation ranges so it pushes the bound
   checks harder, and add an 8B-geometry variant (head dim 128).
3. Move `int8` under an explicit `bitexact.quantized` name, or into the plugin, so the package top level
   holds only the bf16-preserving path.
4. Replace the pickle constants format with safetensors plus a JSON header, so reports and constants can
   be exchanged without trusting pickle.
5. Bring the remaining modules (mostly `vllm_plugin`, `int8`) under the ruff check that CI already runs on
   the core.
