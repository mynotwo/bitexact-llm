"""Llama-3.2-1B geometry, the integer KV cache, and checkpoint helpers shared by both engines."""

import json
import os

import torch

H, HKV, HD, D, FF = 32, 8, 64, 2048, 8192      # query heads, KV heads, head dim, hidden, MLP
V = 128256                                      # vocabulary
EOS, PAD = 128001, 128004


class Cache:
    """Integer K and V per layer, int32 [B, HKV, L, HD] (values within 3 limbs, checked by the model)."""

    def __init__(self, n_layers, B, L, device):
        z = lambda: torch.zeros(B, HKV, L, HD, dtype=torch.int32, device=device)
        self.k = [z() for _ in range(n_layers)]
        self.v = [z() for _ in range(n_layers)]

    def write(self, layer, k, v, pos):
        """k, v int64 [B, T, HKV, HD] at positions pos [B, T]."""
        b = torch.arange(k.shape[0], device=k.device)[:, None].expand_as(pos)
        self.k[layer].permute(0, 2, 1, 3)[b, pos] = k.to(torch.int32)
        self.v[layer].permute(0, 2, 1, 3)[b, pos] = v.to(torch.int32)


def load_checkpoint(model_dir):
    """(config dict, state dict) of a HF Llama directory with a single model.safetensors."""
    from safetensors.torch import load_file
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg, load_file(os.path.join(model_dir, "model.safetensors"))


def rope_inv_freq(model_dir):
    """Llama-3 scaled inverse frequencies (float64, from transformers' own rule)."""
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = LlamaConfig.from_pretrained(model_dir)
    try:
        inv, scale = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cpu")
    except TypeError:                                   # newer transformers signature
        inv, scale = ROPE_INIT_FUNCTIONS["llama3"](cfg, torch.device("cpu"), None)
    assert scale == 1.0
    return inv.double()


def rope_tables(model_dir, n_pos, bits):
    """Integer cos / sin [n_pos, HD / 2] at `bits` fractional bits (float64, rounded once)."""
    ang = torch.arange(n_pos, dtype=torch.float64)[:, None] * rope_inv_freq(model_dir)[None, :]
    return torch.round(torch.cos(ang) * 2.0 ** bits).long(), torch.round(torch.sin(ang) * 2.0 ** bits).long()


def ln_table(bits=40):
    """ln(1 + i / 2^16) at `bits` fractional bits, i = 0..2^16 (integer Gumbel noise)."""
    i = torch.arange((1 << 16) + 1, dtype=torch.float64)
    return torch.round(torch.log1p(i / (1 << 16)) * 2.0 ** bits).long()


def digest(c, skip=("cfg", "stats", "sha256"), name_only=()):
    """SHA-256 over every tensor and scalar of a constants dict, in key order: identifies the model.
    Keys in `skip` are left out; keys in `name_only` contribute their name only."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(c):
        if k in skip:
            continue
        v = c[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode())
            h.update(v.contiguous().numpy().tobytes())
        elif k not in name_only:
            h.update(repr(v).encode())
    return h.hexdigest()
