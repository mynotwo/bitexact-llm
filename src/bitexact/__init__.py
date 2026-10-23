"""bitexact: bit-reproducible Llama-3.2-1B inference with integer arithmetic.

  bitexact.exact        original bf16 weights, int64 fixed-point activations, exact limb int8 GEMMs
  bitexact.int8         int8 weights: faster, quantized integer model
  bitexact.vllm_plugin  the int8 engine inside vLLM (optional extra)
  bitexact.det_float    batch-invariant float bf16 engine (Triton), for comparison
"""

from bitexact.generate import generate, load, logits_sha

__version__ = "0.1.0"
__all__ = ["generate", "load", "logits_sha"]
