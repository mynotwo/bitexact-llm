"""Constants of the int8 engine (Llama-3.2-1B), built once on a CPU from the bf16 checkpoint.

Weights are quantized to int8 (per-output-channel scale), so this mode is an integer model close to, but
not equal to, the bf16 one. Everything is float64 math at build time, rounded once and saved; nothing float
is recomputed at run time. The SHA-256 of the dict (all tensors, in key order) identifies the model.

Contents:
  emb      int32 [V, D]   embedding at FR fractional bits (input side only)
  layer i  wqkv / wo / wgu / wd: int8 weights [N, K], per-output-channel scale s_c = amax_c / 127,
           kept as an integer multiplier m_c = round(s_c / 127 * 2^fm) with one fm per matrix
           (chosen so |acc * m_c| < 2^62 for up to 3 activation limbs)
           ga / gm: RMSNorm gammas at FG bits (int64 [D])
  gf, lm   final RMSNorm gamma and lm_head (= embedding, tied) quantized like the other weights
  exp      e^z at FE bits for z in [EXP_MIN, 0] at step 2^-FA
  silu     SiLU(x) at FR bits for x in [-32, 32] at step 2^-FA (2^18 + 1 entries)
  cos, sin rotary tables at FT bits, [MAX_POS, 32], llama3 frequency scaling
  ln       ln(1 + i / 2^16) at FL bits, i = 0..2^16 (integer Gumbel noise)
"""

import math

import torch

from bitexact.llama import digest, ln_table, load_checkpoint, rope_tables

FR = 16          # residual stream, layer outputs, logits
FG = 12          # RMSNorm gamma
FA = 12          # LUT input step (exp, SiLU)
FE = 30          # exp table values
FT = 16          # cos / sin
FL = 40          # ln table
SILU_LIM = 32
EXP_MIN = -32    # exp table domain [EXP_MIN, 0]
MAX_POS = 2048
QMAX_BITS = 23   # worst-case activation magnitude used to pick fm (3 limbs: 127 * 2^16 < 2^23)


def _q_linear(w):
    """float64 [N, K] -> (int8 [N, K], int64 multipliers [N], fm)."""
    amax = w.abs().amax(1).clamp_min(1e-30)
    s = amax / 127.0
    q = torch.round(w / s[:, None]).clamp(-127, 127)
    l1 = q.abs().sum(1)                                     # sum_k |w_int|
    worst = float((l1 * s / 127.0).max()) * 2.0 ** QMAX_BITS * 127.0
    fm = min(60, int(math.floor(62 - math.log2(worst))) - 1)
    m = torch.round(s / 127.0 * 2.0 ** fm).long()
    return q.to(torch.int8), m, fm


def exp_table():
    """e^x at FE bits for x = EXP_MIN .. 0 at step 2^-FA; index x * 2^FA - (EXP_MIN << FA)."""
    xs = torch.arange(EXP_MIN << FA, 1, dtype=torch.float64) / (1 << FA)
    return torch.round(torch.exp(xs) * (1 << FE)).long()


def build(model_dir, n_layers=None):
    """Constants dict from a HF Llama-3.2-1B directory; c["sha256"] identifies it."""
    cfg, sd = load_checkpoint(model_dir)
    g = lambda k: sd[k].double()
    L = cfg["num_hidden_layers"] if n_layers is None else n_layers
    c = {"n_layers": L, "cfg": cfg}
    emb = g("model.embed_tokens.weight")
    assert float(emb.abs().max()) < 2 ** (31 - FR)
    c["emb"] = torch.round(emb * (1 << FR)).to(torch.int32)
    for i in range(L):
        p = f"model.layers.{i}."
        wqkv = torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
        wgu = torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])
        for name, w in [("wqkv", wqkv), ("wo", g(p + "self_attn.o_proj.weight")), ("wgu", wgu),
                        ("wd", g(p + "mlp.down_proj.weight"))]:
            q, m, fm = _q_linear(w)
            c[f"{i}.{name}"], c[f"{i}.{name}.m"], c[f"{i}.{name}.fm"] = q, m, fm
        c[f"{i}.ga"] = torch.round(g(p + "input_layernorm.weight") * (1 << FG)).long()
        c[f"{i}.gm"] = torch.round(g(p + "post_attention_layernorm.weight") * (1 << FG)).long()
    c["gf"] = torch.round(g("model.norm.weight") * (1 << FG)).long()
    lm = g("lm_head.weight") if "lm_head.weight" in sd else emb
    c["lm"], c["lm.m"], c["lm.fm"] = _q_linear(lm)
    c["exp"] = exp_table()
    xs = torch.arange(-SILU_LIM << FA, (SILU_LIM << FA) + 1, dtype=torch.float64) / (1 << FA)
    c["silu"] = torch.round(xs / (1.0 + torch.exp(-xs)) * (1 << FR)).long()
    c["cos"], c["sin"] = rope_tables(model_dir, MAX_POS, FT)
    c["ln"] = ln_table(FL)
    for k in c:
        if k.endswith(".ga") or k.endswith(".gm") or k == "gf":
            assert int(c[k].abs().max()) < 1 << 20, k
    c["sha256"] = sha256(c)
    return c


def sha256(c):
    """The model's identity: SHA-256 of every tensor and scalar (config: key name only)."""
    return digest(c, skip=("sha256",), name_only=("cfg",))
