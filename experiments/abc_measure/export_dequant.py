"""Export the BitExact int8 Llama-3.2-1B constants as a dequantized HF checkpoint (fp32 safetensors).

  python export_dequant.py --const C.pt --orig HF_DIR --out OUT_DIR

Weight = q * s per output row, s = m / 2^fm * 127 (m, fm: the engine's integer multiplier); gammas =
integer / 2^12; embedding = integer / 2^16. The tied head becomes a separate quantized lm_head
(tie_word_embeddings = false). Prints exactness statistics: whether q * s is representable in bf16 / fp32,
and the distance to the original weights (in units of the row scale s; a correct dequantization is <= 0.5).
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SPLITS = {"wqkv": [("self_attn.q_proj", 2048), ("self_attn.k_proj", 512), ("self_attn.v_proj", 512)],
          "wo": [("self_attn.o_proj", 2048)],
          "wgu": [("mlp.gate_proj", 8192), ("mlp.up_proj", 8192)],
          "wd": [("mlp.down_proj", 2048)]}
FG, FR = 12, 16


def deq(q, m, fm):
    s = m.double() / 2.0 ** fm * 127.0
    return q.double() * s[:, None], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    orig = load_file(os.path.join(a.orig, "model.safetensors"))
    sd, st = {}, {"n": 0, "fp32_exact": 0, "bf16_exact": 0, "max_rel_fp32": 0.0, "max_dev_over_s": 0.0}

    def put(name, w64, ref=None, s=None):
        w32 = w64.float()
        st["n"] += w64.numel()
        st["fp32_exact"] += int((w32.double() == w64).sum())
        st["bf16_exact"] += int((w64.bfloat16().double() == w64).sum())
        nz = w64 != 0
        st["max_rel_fp32"] = max(st["max_rel_fp32"], float(((w32.double() - w64).abs()[nz] / w64.abs()[nz]).max()))
        if ref is not None:
            dev = ((w64 - ref.double()).abs() / s[:, None]).max()
            st["max_dev_over_s"] = max(st["max_dev_over_s"], float(dev))
        sd[name] = w32.contiguous()

    for i in range(c["n_layers"]):
        p = f"model.layers.{i}."
        for k, parts in SPLITS.items():
            w, s = deq(c[f"{i}.{k}"], c[f"{i}.{k}.m"], c[f"{i}.{k}.fm"])
            o = 0
            for nm, n in parts:
                ref = orig[p + nm + ".weight"]
                put(p + nm + ".weight", w[o:o + n], ref, s[o:o + n])
                o += n
        sd[p + "input_layernorm.weight"] = (c[f"{i}.ga"].double() / 2 ** FG).float()
        sd[p + "post_attention_layernorm.weight"] = (c[f"{i}.gm"].double() / 2 ** FG).float()
    sd["model.norm.weight"] = (c["gf"].double() / 2 ** FG).float()
    sd["model.embed_tokens.weight"] = (c["emb"].double() / 2 ** FR).float()
    w, s = deq(c["lm"], c["lm.m"], c["lm.fm"])
    put("lm_head.weight", w, orig["model.embed_tokens.weight"], s)
    os.makedirs(a.out, exist_ok=True)
    save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(a.orig, "config.json")))
    cfg.update(tie_word_embeddings=False, torch_dtype="float32")
    cfg.pop("dtype", None)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.orig):
        if f not in ("config.json", "model.safetensors") and not os.path.exists(os.path.join(a.out, f)):
            shutil.copy(os.path.realpath(os.path.join(a.orig, f)), os.path.join(a.out, f))
    rep = {"n_quantized_weights": st["n"], "frac_fp32_exact": st["fp32_exact"] / st["n"],
           "frac_bf16_exact": st["bf16_exact"] / st["n"], "max_rel_err_fp32_round": st["max_rel_fp32"],
           "max_abs_dev_from_orig_over_s": st["max_dev_over_s"], "const_sha": c.get("sha256")}
    json.dump(rep, open(os.path.join(a.out, "export_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
