"""bitexact: bit-reproducible Llama inference with integer arithmetic.

  bitexact.exact        the engine: original bf16 weights, int64 fixed-point activations, exact limb GEMMs
  bitexact.repro        identity reports: self-check, record, compare
  bitexact.synthetic    platform-independent synthetic checkpoint for weight-free whole-model checks
  bitexact.vllm_plugin  the integer engines inside vLLM (optional extra)
  bitexact.int8         int8 weights: faster, but a quantized (different) model
"""

from bitexact.generate import generate, load, logits_sha

__version__ = "0.1.0"
__all__ = ["generate", "load", "logits_sha"]
