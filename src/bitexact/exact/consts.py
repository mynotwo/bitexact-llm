"""Constants of the exact-bf16 engine (Llama-3.2-1B), built once on a CPU from the bf16 checkpoint.

Every bf16 weight w (sign s, 8-bit significand mag, exponent ex: w = s * mag * 2^ex, ex already includes
the -7 of the significand) is split, per output row with one grid exponent g, into
  dense part   W = s * trunc(mag * 2^(ex - g)), |W| <= LIM3 = 8355711, stored as 3 signed int8 limbs
  residual     R = w - W * 2^g, nonzero only where the row's span exceeds 23 bits (~5e-5 of entries);
               stored sparse as integers on a finer row grid gf = g - d (R_int * 2^gf)
g is the smallest exponent with max |W| <= LIM3. decompose() checks W * 2^g + R_int * 2^gf == w bit for bit
(float64 holds both sides exactly), so every linear layer, including the tied lm_head, uses the original
bf16 weights with zero loss.

RMSNorm gammas keep their bf16 value exactly as (s * mag, ex) per element. The input embedding table is
the one exception: it is stored at the residual stream's fixed point (int32, unit 2^-FR = 2^-24), i.e. each
entry is rounded once to 2^-24 (|error| <= 2^-25; ~0.02% of Llama-3.2-1B's entries are affected).
Tables (RoPE, 2^-f, ln) are computed in float64 and stored; the SHA-256 of the dict identifies the model.
"""

import torch

from bitexact.intops import LIM, split_limbs
from bitexact.llama import digest, ln_table, load_checkpoint, rope_tables

FR = 24                       # residual stream, linear outputs, logits
FT = 30                       # cos / sin
FL = 40                       # ln table (integer Gumbel noise)
EXP2_BITS = 14                # 2^-f table: f step 2^-14, linear interpolation
EXP2_ONE = 60                 # table values at 2^60
MAX_POS = 4096
LIM3 = LIM[3]                 # 3 signed base-256 digits
NW = 3                        # weight limbs


def bf16_parts(w):
    """bf16 tensor -> (signed significand int64, exponent int64): w = sm * 2^ex exactly."""
    b = w.contiguous().view(torch.int16).long() & 0xFFFF
    e = (b >> 7) & 0xFF
    assert not bool((e == 255).any()), "inf/nan weight"
    m = b & 0x7F
    mag = torch.where(e > 0, m | 0x80, m)
    ex = torch.where(e > 0, e, torch.ones_like(e)) - 134
    sm = torch.where((b >> 15) == 1, -mag, mag)
    return sm, ex


def _bitlen(x):
    """Bit length of 0 <= x < 2^9 (significands)."""
    out = torch.zeros_like(x)
    for i in range(9):
        out = torch.where(x >> i != 0, torch.full_like(x, i + 1), out)
    return out


def _lowbit(x):
    """Index of the lowest set bit of 0 < x < 2^9; 99 for x == 0."""
    out = torch.full_like(x, 99)
    for i in range(8, -1, -1):
        out = torch.where(((x >> i) & 1) == 1, torch.full_like(x, i), out)
    return out


def _limbs(W):
    out, r = split_limbs(W, NW)
    assert not bool(r.any()), "limb split left a remainder"
    return torch.stack([lo.to(torch.int8) for lo in out])


def decompose(w):
    """bf16 [N, K] -> dict(limbs int8 [3, N, K], g [N], d [N], rr/rc/rv residual COO, stats)."""
    sm, ex = bf16_parts(w)
    mag = sm.abs()
    nz = mag > 0
    top = torch.where(nz, ex + _bitlen(mag), torch.full_like(ex, -10 ** 6)).amax(1)
    g = top - 23
    while True:
        sh = (ex - g[:, None]).clamp(max=40)
        big = nz & (sh >= 0) & ((mag << sh.clamp(min=0)) > LIM3)
        bad = big.any(1)
        if not bool(bad.any()):
            break
        g = torch.where(bad, g + 1, g)
    g = torch.where(top < -10 ** 5, torch.zeros_like(g), g)         # all-zero rows
    sh = ex - g[:, None]
    up = sh >= 0
    dmag = torch.where(up, mag << sh.clamp(min=0, max=40), mag >> (-sh).clamp(min=0, max=40))
    rmag = mag - torch.where(up, mag, dmag << (-sh).clamp(min=0, max=40))   # low bits dropped by trunc
    rmag = torch.where(up, torch.zeros_like(rmag), rmag)
    W = torch.where(sm < 0, -dmag, dmag)
    has = rmag > 0
    low = torch.where(has, ex + _lowbit(rmag), torch.full_like(ex, 10 ** 6))
    gf = torch.minimum(low.amin(1), g)
    d = g - gf
    assert int(d.max()) <= 30, "residual grid too fine"
    rr, rc = has.nonzero(as_tuple=True)
    rsh_ = ex[rr, rc] - gf[rr]
    rm_ = rmag[rr, rc]
    rv = torch.where(rsh_ >= 0, rm_ << rsh_.clamp(min=0), rm_ >> (-rsh_).clamp(min=0))
    rv = torch.where(sm[rr, rc] < 0, -rv, rv)
    # exactness: W * 2^g + R * 2^gf == w (float64 is exact here: <= 2^53 integers times powers of 2)
    rec = W.double() * torch.pow(2.0, g.double())[:, None]
    rec[rr, rc] += rv.double() * torch.pow(2.0, gf[rr].double())
    same_val = bool((rec == w.double()).all())
    same_bits = bool((rec.to(torch.bfloat16).view(torch.int16) == w.view(torch.int16)).all())
    negzero = int(((w.view(torch.int16).long() & 0xFFFF) == 0x8000).sum())
    assert same_val, "reconstruction differs"
    return {"limbs": _limbs(W), "g": g, "d": d, "rr": rr, "rc": rc, "rv": rv,
            "stats": {"same_val": same_val, "same_bits": same_bits, "negzero": negzero, "nnz_res": len(rr),
                      "rows_res": int(has.any(1).sum()), "max_d": int(d.max()), "maxW": int(W.abs().max())}}


def tables(model_dir):
    t = dict(zip(("cos", "sin"), rope_tables(model_dir, MAX_POS, FT)))
    f = torch.arange((1 << EXP2_BITS) + 1, dtype=torch.float64) / (1 << EXP2_BITS)
    t["exp2"] = torch.round(torch.pow(2.0, -f) * 2.0 ** EXP2_ONE).long()
    t["ln"] = ln_table(FL)
    return t


def build(model_dir, n_layers=None):
    """Constants dict from a HF Llama-3.2-1B directory; c["sha256"] identifies it."""
    cfg, sd = load_checkpoint(model_dir)
    L = cfg["num_hidden_layers"] if n_layers is None else n_layers
    c = {"n_layers": L, "cfg": cfg, "stats": {}}

    def put(name, w):
        dec = decompose(w)
        c["stats"][name] = dec.pop("stats")
        for k, v in dec.items():
            c[f"{name}.{k}"] = v

    emb = sd["model.embed_tokens.weight"]
    sm, ex = bf16_parts(emb)
    s = ex + FR
    e = torch.where(s >= 0, sm << s.clamp(min=0), (sm + (1 << (-s - 1).clamp(min=0, max=62))) >> (-s).clamp(max=63))
    assert int(e.abs().max()) < 1 << 31
    c["emb"] = e.to(torch.int32)
    for i in range(L):
        p = f"model.layers.{i}."
        cat = lambda *ks: torch.cat([sd[p + k] for k in ks])
        put(f"{i}.wqkv", cat("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"))
        put(f"{i}.wo", sd[p + "self_attn.o_proj.weight"])
        put(f"{i}.wgu", cat("mlp.gate_proj.weight", "mlp.up_proj.weight"))
        put(f"{i}.wd", sd[p + "mlp.down_proj.weight"])
        for n, k in (("ga", "input_layernorm.weight"), ("gm", "post_attention_layernorm.weight")):
            c[f"{i}.{n}.m"], c[f"{i}.{n}.e"] = bf16_parts(sd[p + k])
    c["gf.m"], c["gf.e"] = bf16_parts(sd["model.norm.weight"])
    put("lm", sd.get("lm_head.weight", emb))
    c.update(tables(model_dir))
    c["sha256"] = sha256(c)
    return c


def sha256(c):
    """The model's identity: SHA-256 of every tensor and scalar except the config and build stats."""
    return digest(c)


def summary(c):
    """Bit-exactness record of the weight decomposition."""
    st = c["stats"]
    return {"layers": c["n_layers"], "sha256": c["sha256"],
            "all_same_val": all(s["same_val"] for s in st.values()),
            "all_same_bits": all(s["same_bits"] for s in st.values()),
            "negzero": sum(s["negzero"] for s in st.values()),
            "nnz_res": sum(s["nnz_res"] for s in st.values()),
            "rows_res": sum(s["rows_res"] for s in st.values()),
            "max_d": max(s["max_d"] for s in st.values())}
