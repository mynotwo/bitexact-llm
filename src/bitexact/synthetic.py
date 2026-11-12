"""A small Llama checkpoint generated from integers, so the full model path can be checked without weights.

Every bf16 weight is assembled bit by bit from numpy PCG64 raw words (sign, exponent, significand), so the
checkpoint is the same bytes on every platform and numpy version: no floating-point random number
generator, no Box-Muller, no libm. A few weights per matrix sit far below the row maximum, which forces the
exact engine's sparse residual path. The geometry (hidden 512, GQA 8/2 heads, head dim 64, two layers,
vocabulary 512) keeps a CPU build and a 16-token generation to a few seconds.

The constants built from it still pass through float64 table generation (RoPE, 2^-f, ln), as for a real
checkpoint; their SHA-256 is part of the identity report, so two machines whose tables differ are reported
as a contract mismatch rather than as a logits difference.
"""

import json
import os

import numpy as np
import torch

SEED = 20261112
CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "torch_dtype": "bfloat16",
    "vocab_size": 512, "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": False, "hidden_act": "silu",
    "attention_bias": False, "mlp_bias": False, "bos_token_id": 1, "eos_token_id": 2,
    "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                     "original_max_position_embeddings": 8192},
}
# Prompts in the synthetic vocabulary (token ids < 512).
PROMPTS = [[1, 17, 301, 45, 222, 9, 88, 410], [1, 500, 3, 3, 71], [1, 260, 129, 64, 32, 16, 8, 4, 2, 255, 511],
           [1, 42]]


def _bf16(rng, shape, lo, hi, tiny=0.0):
    """bf16 tensor with exponents uniform in [lo, hi] (unbiased powers of two), random sign and significand;
    a fraction `tiny` of entries get exponent lo - 20 (below a 23-bit row span)."""
    n = int(np.prod(shape))
    w = rng.bit_generator.random_raw(n).astype(np.uint64)
    sign = (w & np.uint64(1)).astype(np.int64)
    mant = ((w >> np.uint64(1)) & np.uint64(0x7F)).astype(np.int64)
    span = hi - lo + 1
    ex = lo + ((w >> np.uint64(8)) & np.uint64(0xFFFF)).astype(np.int64) % span
    small = ((w >> np.uint64(24)) & np.uint64(0xFFFFFF)).astype(np.int64) < int(tiny * (1 << 24))
    ex = np.where(small, lo - 20, ex)
    bits = (sign << 15) | ((ex + 127) << 7) | mant
    return torch.from_numpy(bits.astype(np.uint16).view(np.int16).reshape(shape).copy()).view(torch.bfloat16)


def state_dict(seed=SEED):
    """name -> bf16 tensor, a pure function of seed."""
    c = CONFIG
    D, F, V, HD = c["hidden_size"], c["intermediate_size"], c["vocab_size"], c["head_dim"]
    H, HKV = c["num_attention_heads"], c["num_key_value_heads"]
    rng = np.random.Generator(np.random.PCG64(seed))
    sd = {"model.embed_tokens.weight": _bf16(rng, (V, D), -8, -3)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name, shape in (("self_attn.q_proj", (H * HD, D)), ("self_attn.k_proj", (HKV * HD, D)),
                            ("self_attn.v_proj", (HKV * HD, D)), ("self_attn.o_proj", (D, H * HD)),
                            ("mlp.gate_proj", (F, D)), ("mlp.up_proj", (F, D)), ("mlp.down_proj", (D, F))):
            sd[p + name + ".weight"] = _bf16(rng, shape, -9, -4, tiny=1e-3)
        sd[p + "input_layernorm.weight"] = _bf16(rng, (D,), -1, 0)
        sd[p + "post_attention_layernorm.weight"] = _bf16(rng, (D,), -1, 0)
    sd["model.norm.weight"] = _bf16(rng, (D,), -1, 0)
    sd["lm_head.weight"] = _bf16(rng, (V, D), -7, -3, tiny=1e-3)
    return {k: v.abs() if k.endswith("norm.weight") else v for k, v in sd.items()}


def write_checkpoint(path, seed=SEED):
    """Write config.json and model.safetensors into directory `path`; returns path."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=1, sort_keys=True)
    save_file({k: v.contiguous() for k, v in state_dict(seed).items()}, os.path.join(path, "model.safetensors"))
    return path


def constants(seed=SEED):
    """Exact-mode constants of the synthetic checkpoint (built in a temporary directory)."""
    import tempfile

    from bitexact.exact.consts import build
    with tempfile.TemporaryDirectory() as d:
        return build(write_checkpoint(d, seed))
