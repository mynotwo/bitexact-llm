"""Tables for RESULTS.md from the prof_scorer / prof_decode / nsys outputs in OUT_DIR.

  python summarize.py OUT_DIR > tables.md
"""
import collections
import glob
import json
import os
import sys


CATS = ["int GEMM, linear layers", "int GEMM, lm_head", "attention QK (bmm)", "attention PV (bmm)",
        "attention fused paged (QK+softmax+PV)", "attention softmax (mask, max, shift, sum)", "exp LUT",
        "attention num/den divide (compiled: + quantization)", "RMSNorm", "RMSNorm + quantization (compiled segment)", "quantization (quant_rows, limb split)", "SiLU LUT", "SiLU x up multiply (compiled: + SiLU LUT + quantization)",
        "GEMM epilogue (limb join, rescale, residual add)", "RoPE / q,k,v split", "other elementwise",
        "log-softmax float64 (scorer)", "sampler (Gumbel, argmax)"]


def category(path):
    p = path.split("/")
    last = p[-1]
    if last == "gemm":
        return "int GEMM, lm_head" if "lm_head" in p else "int GEMM, linear layers"
    if last == "attn_qk":
        return "attention QK (bmm)"
    if last == "attn_pv":
        return "attention PV (bmm)"
    if "attn_paged" in p:
        return "attention fused paged (QK+softmax+PV)"
    if last == "exp_lut":
        return "exp LUT"
    if last == "softmax":
        return "attention softmax (mask, max, shift, sum)"
    if last in ("quant", "limb_split"):
        return "quantization (quant_rows, limb split)"
    if last == "attn_out":
        return "attention num/den divide (compiled: + quantization)"
    if last == "pre_plain":
        return "quantization (quant_rows, limb split)"
    if last == "silu_mul":
        return "SiLU x up multiply (compiled: + SiLU LUT + quantization)"
    if last == "pre_norm":
        return "RMSNorm + quantization (compiled segment)" if "rmsnorm" not in p else "RMSNorm"
    if last == "rmsnorm":
        return "RMSNorm"
    if last == "silu_lut":
        return "SiLU LUT"
    if last in ("post", "limb_join"):
        return "GEMM epilogue (limb join, rescale, residual add)"
    if last in ("rope", "qkv_rope"):
        return "RoPE / q,k,v split"
    if last == "logp":
        return "log-softmax float64 (scorer)"
    if "sampler" in p:
        return "sampler (Gumbel, argmax)"
    return "other elementwise"          # kv_write, overflow flags, embedding, cat, prefill / lm_head glue


def by_cat(by_path, idx=0):
    out = collections.Counter()
    for k, v in by_path.items():
        out[category(k)] += v[idx] if isinstance(v, list) else v
    return out


def time_table(title, by_path, busy_us, wall_us, tokens=None, scale=1.0):
    """by_path values [us, count]; scale converts us to the unit shown (ms)."""
    t = by_cat(by_path, 0)
    n = by_cat(by_path, 1)
    tot = sum(t.values())
    print(f"\n**{title}**\n")
    print("| category | kernel ms | kernels | % of GPU kernel time | % of wall |")
    print("|---|---|---|---|---|")
    for c in [c for c in CATS if c in t]:
        print(f"| {c} | {t[c] * scale:.2f} | {n[c]:.0f} | {100 * t[c] / tot:.1f} | {100 * t[c] / wall_us:.1f} |")
    print(f"| **sum of kernel time** | {tot * scale:.2f} | {sum(n.values()):.0f} | 100 | {100 * tot / wall_us:.1f} |")
    print(f"| GPU busy (union of kernels) | {busy_us * scale:.2f} | | | {100 * busy_us / wall_us:.1f} |")
    print(f"| GPU idle: launch / Python / host sync | {(wall_us - busy_us) * scale:.2f} | | | "
          f"{100 * (wall_us - busy_us) / wall_us:.1f} |")
    print(f"| wall | {wall_us * scale:.2f} | | | 100 |")
    return t


def class_table(title, by_class, busy_us, wall_us, scale=1.0):
    tot = sum(v[0] for v in by_class.values())
    print(f"\n**{title}**\n")
    print("| kernel class | kernel ms | kernels | % of GPU kernel time | % of wall |")
    print("|---|---|---|---|---|")
    for k, (us, cnt) in sorted(by_class.items(), key=lambda x: -x[1][0]):
        print(f"| {k} | {us * scale:.2f} | {cnt:.0f} | {100 * us / tot:.1f} | {100 * us / wall_us:.1f} |")
    print(f"| GPU busy | {busy_us * scale:.2f} | | | {100 * busy_us / wall_us:.1f} |")
    print(f"| GPU idle | {(wall_us - busy_us) * scale:.2f} | | | {100 * (wall_us - busy_us) / wall_us:.1f} |")


def gemm_rows(rows, title):
    print(f"\n**{title}** (H100 SXM dense int8 peak 1979 TOPS; one `torch._int_mm` = nl limbs stacked as rows)\n")
    print("| M (= nl x rows, padded) | N | K | lm_head | calls | us each | TOPS | % of peak |")
    print("|---|---|---|---|---|---|---|---|")
    for g in rows[:10]:
        print(f"| {g['M']} | {g['N']} | {g['K']} | {'y' if g['lm_head'] else ''} | {g['count']} | {g['us_each']} | "
              f"{g['TOPS']} | {100 * g['TOPS'] / 1979:.0f} |")


def scorer(d):
    for f in sorted(x for x in glob.glob(f"{d}/sc_*.json") if ".trace." not in x):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        p = r["profile"]
        print(f"\n### {name}: {r['args']['rollout'].split('/')[-1]}, batch {r['args']['bs']}, "
              f"{'compiled' if r['args']['compiled'] else 'eager'}, nl {r['args']['nl']}")
        print(f"\nunprofiled: {r['timing']['tok_per_s']} tok/s over {r['timing']['batches']} batches "
              f"({r['timing']['seq_tokens']} tokens, sec {r['timing']['sec']}); padding: {r['padding']}")
        wall = p["wall_s_profiled"] * 1e6
        print(f"profiled: {p['batches']} batches, {p['seq_tokens']} tokens, wall {p['wall_s_profiled']} s, "
              f"{p['n_kernels']} kernels ({p['n_kernels'] / p['seq_tokens']:.1f} per token)")
        if True:
            time_table(f"{name}: time by category (ms over the profiled batches)", p["by_path"], p["busy_us"], wall,
                       scale=1e-3)
        class_table(f"{name}: time by kernel class", p["by_kernel_class"], p["busy_us"], wall, scale=1e-3)
        gemm_rows(r["gemm"], f"{name}: int8 GEMMs")
        if "bytes" in r:
            b = r["bytes"]
            bc = by_cat({k: v for k, v in b["by_path"].items()})
            tot = sum(bc.values())
            print(f"\n**{name}: logical bytes per token** (one batch, {b['batch_tokens_real']} real / "
                  f"{b['batch_tokens_padded']} padded tokens; eager ops' inputs + outputs, Triton bmm inputs + output)\n")
            print("| category | MB per real token | % |")
            print("|---|---|---|")
            for c in [c for c in CATS if c in bc]:
                print(f"| {c} | {bc[c] / b['batch_tokens_real'] / 1e6:.2f} | {100 * bc[c] / tot:.1f} |")
            print(f"| **total** | {tot / b['batch_tokens_real'] / 1e6:.2f} | 100 |")


def decode(d):
    for f in sorted(x for x in glob.glob(f"{d}/dec_*.json") if ".trace." not in x):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        p = r["profile"]
        ms = {k: v for k, v in r.items() if k.startswith("ms")}
        print(f"\n### {name}: B {r['args']['B']}, ctx {r['args']['ctx']}, {r['args']['mode']}\n\n{ms}; "
              f"profiled step: wall {p['wall_ms_per_step']} ms, busy {p['busy_ms_per_step']} ms, "
              f"{p['kernels_per_step']:.0f} kernels")
        wall = p["wall_ms_per_step"] * 1e3
        if r["args"]["mode"] == "eager":
            time_table(f"{name}: time by category per step (ms)", p["by_path_us_per_step"], p["busy_ms_per_step"] * 1e3,
                       wall, scale=1e-3)
        class_table(f"{name}: time by kernel class per step", p["by_kernel_class_us_per_step"],
                    p["busy_ms_per_step"] * 1e3, wall, scale=1e-3)
        gemm_rows([dict(g, us_each=g["us_each"]) for g in r["gemm"]], f"{name}: int8 GEMMs")
        if "bytes" in r:
            bc = by_cat(r["bytes"]["by_path"])
            bc["attention fused paged (QK+softmax+PV)"] += r["bytes"]["attn_paged_kv_read_est"]
            tot = sum(bc.values())
            B = r["args"]["B"]
            print(f"\n**{name}: logical bytes per token** (paged attention: K read twice + V once, int32)\n")
            print("| category | MB per token | % |")
            print("|---|---|---|")
            for c in [c for c in CATS if c in bc]:
                print(f"| {c} | {bc[c] / B / 1e6:.2f} | {100 * bc[c] / tot:.1f} |")
            print(f"| **total** | {tot / B / 1e6:.2f} | 100 |")


def vllm(d):
    for f in sorted(glob.glob(f"{d}/vllm_B*.json")):
        r = json.load(open(f))
        B = r["B"]
        print(f"\n### vLLM plugin decode B {B}, {r['gen']} new tokens: {r['tok_per_s_median']} tok/s, "
              f"{r['ms_per_step_median']} ms per step (wall, median of 3)")
        p = r["profile"]
        class_table(f"vLLM B {B}: kernel classes over one profiled generate of {p['gen']} new tokens "
                    f"({p['n_kernels']} kernels, {p['n_kernels'] / p['gen']:.0f} per step; wall {p['wall_us'] / 1e3:.1f} ms)",
                    p["by_kernel_class"], p["busy_us"], p["wall_us"], scale=1e-3)


def share_table(d, runs):
    """Combined table: % of wall per category for several scorer runs."""
    cols = {}
    for r in runs:
        x = json.load(open(f"{d}/{r}.json"))
        p = x["profile"]
        cols[r] = (by_cat(p["by_path"], 0), p["wall_s_profiled"] * 1e6, p["busy_us"], x["timing"]["tok_per_s"])
    cats = [c for c in CATS if any(c in cols[r][0] for r in runs)]
    print("| category (% of wall) | " + " | ".join(runs) + " |")
    print("|---" * (len(runs) + 1) + "|")
    for c in cats:
        print(f"| {c} | " + " | ".join(f"{100 * cols[r][0].get(c, 0) / cols[r][1]:.1f}" for r in runs) + " |")
    print("| GPU idle (launch, Python, host sync) | " +
          " | ".join(f"{100 * (cols[r][1] - cols[r][2]) / cols[r][1]:.1f}" for r in runs) + " |")
    print("| scorer tok/s (unprofiled) | " + " | ".join(f"{cols[r][3]:,.0f}" for r in runs) + " |")


if __name__ == "__main__":
    d = sys.argv[1]
    print("## Scorer")
    scorer(d)
    print("\n## Decode step (outside vLLM)")
    decode(d)
    print("\n## vLLM decode (torch.profiler)")
    vllm(d)
    print("\n## Combined scorer shares")
    share_table(d, ["sc_gsm_b8_eager", "sc_gsm_b8_comp", "sc_4k_b4_eager", "sc_4k_b4_comp", "sc_4k_b1_eager", "sc_4k_b1_comp"])
