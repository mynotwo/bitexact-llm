"""Plots and tables for the A/B/C RL runs.

  python analyze.py --runs RUNS_DIR --out FIG_DIR      (run dirs named {arm}_s{seed})

Writes reward.png, mismatch.png, eval.png, stability.png and prints markdown tables.
"""

import argparse
import glob
import json
import os
import statistics as stt

ARMS = ["A", "A_TIS", "B", "C"]
COL = {"A": "#1f77b4", "A_TIS": "#17becf", "B": "#ff7f0e", "C": "#2ca02c"}


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def load(runs):
    out = {}
    for d in sorted(x for x in glob.glob(os.path.join(runs, "*_s*")) if os.path.isdir(x)):
        name = os.path.basename(d)
        arm, seed = name.rsplit("_s", 1)
        if arm in ARMS:
            steps = [r for r in jl(os.path.join(d, "steps.jsonl")) if "reward" in r]
            out[(arm, int(seed))] = {"steps": steps, "eval": jl(os.path.join(d, "eval.jsonl")),
                                     "bound": [r for r in jl(os.path.join(d, "steps.jsonl")) if "bound_error" in r]}
    return out


def smooth(x, k=10):
    return [sum(x[max(0, i - k + 1):i + 1]) / len(x[max(0, i - k + 1):i + 1]) for i in range(len(x))]


def ms(v):
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{stt.mean(v):.3f} (" + ", ".join(f"{x:.3f}" for x in v) + ")"


def plots(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    panels = [("reward.png", [("reward", lambda r: r["reward"], "train reward (10-step mean)", True)]),
              ("mismatch.png", [("mean |Δ|", lambda r: r["delta"]["mean_abs"], "mean |log p_trainer − log p_sampler|", False),
                                ("clip", lambda r: r["clip_frac"], "PPO clip-active token fraction", False),
                                ("kl", lambda r: r["kl_k3"], "KL(sampler‖trainer), k3", False)]),
              ("stability.png", [("grad", lambda r: r["grad_norm"], "grad norm (pre-clip)", False),
                                 ("len", lambda r: r["resp_len"], "mean response tokens", False),
                                 ("chg", lambda r: r["int8_changed_frac"], "int8 entries changed vs previous step", False)])]
    for fn, ps in panels:
        fig, axs = plt.subplots(1, len(ps), figsize=(5 * len(ps), 3.6), squeeze=False)
        for ax, (_, f, title, sm) in zip(axs[0], ps):
            for (arm, seed), d in sorted(R.items()):
                xs = [r["step"] for r in d["steps"]]
                ys = [f(r) for r in d["steps"]]
                ax.plot(xs, smooth(ys) if sm else ys, color=COL[arm], lw=1.2,
                        ls="-" if seed == 1 else "--", label=f"{arm} s{seed}")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step")
            if fn == "mismatch.png" and title.startswith("mean"):
                ax.set_yscale("symlog", linthresh=1e-6)
        axs[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fn), dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axs, ["int_acc", "float_acc"]):
        for (arm, seed), d in sorted(R.items()):
            ev = [e for e in d["eval"] if key in e]
            ax.plot([e["step"] for e in ev], [e[key] for e in ev], marker="o", ms=3, color=COL[arm],
                    ls="-" if seed == 1 else "--", label=f"{arm} s{seed}")
        ax.set_title(f"GSM8K test[:500] greedy, {'integer engine' if key == 'int_acc' else 'vLLM bf16'}", fontsize=10)
        ax.set_xlabel("step")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eval.png"), dpi=130)
    plt.close(fig)


def tables(R):
    seeds = sorted({s for _, s in R})
    print("| arm | int engine acc, step 0 | int engine acc, final | vLLM bf16 acc, final | train reward, last 20 steps |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        rs = [R[(arm, s)] for s in seeds if (arm, s) in R]
        if not rs:
            continue
        e0 = [d["eval"][0]["int_acc"] for d in rs if d["eval"]]
        ei = [d["eval"][-1]["int_acc"] for d in rs if d["eval"]]
        ef = [d["eval"][-1]["float_acc"] for d in rs if d["eval"]]
        tr = [stt.mean(r["reward"] for r in d["steps"][-20:]) for d in rs if d["steps"]]
        print(f"| {arm} | {ms(e0)} | {ms(ei)} | {ms(ef)} | {ms(tr)} |")
    print()
    print("| arm | steps | mean \\|Δ\\| | max \\|Δ\\| | tokens Δ≠0 | ratio outside [0.8,1.2] | clip-active | KL k3 | grad norm max | non-finite grads | bound errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        st = [r for s in seeds if (arm, s) in R for r in R[(arm, s)]["steps"]]
        if not st:
            continue
        nb = sum(len(R[(arm, s)]["bound"]) for s in seeds if (arm, s) in R)
        f = lambda k: stt.mean(r[k] for r in st)
        print(f"| {arm} | {len(st)} | {stt.mean(r['delta']['mean_abs'] for r in st):.2e} | "
              f"{max(r['delta']['max_abs'] for r in st):.3g} | {stt.mean(r['delta']['frac_nonzero'] for r in st):.4f} | "
              f"{f('ratio_outside_clip'):.2e} | {f('clip_frac'):.2e} | {f('kl_k3'):.2e} | "
              f"{max(r['grad_norm'] for r in st):.3f} | {sum(not r['grad_finite'] for r in st)} | {nb} |")
    print()
    print("| arm | s/step (median) | sample | int rescoring | trainer fwd+bwd | quantize+sync | peak mem GB |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        st = [r for s in seeds if (arm, s) in R for r in R[(arm, s)]["steps"]]
        if not st:
            continue
        md = lambda f: stt.median(f(r) for r in st)
        print(f"| {arm} | {md(lambda r: r['step_sec']):.1f} | {md(lambda r: r['time']['sample']):.1f} | "
              f"{md(lambda r: r['time']['score_int']):.1f} | {md(lambda r: r['time']['backward']):.1f} | "
              f"{md(lambda r: r['time']['quantize']):.2f} | {max(r['mem_gb'] for r in st):.1f} |")
    ev = [(arm, e) for (arm, _), d in R.items() for e in d["eval"] if "int_sec" in e]
    if ev:
        print(f"\neval (500 greedy): integer engine {stt.median(e['int_sec'] for _, e in ev):.0f} s, "
              f"vLLM bf16 {stt.median(e['float_sec'] for _, e in ev):.1f} s (median)")
    c = [r["float_vs_int_trainer"]["mean_abs"] for s in seeds if ("C", s) in R for r in R[("C", s)]["steps"]
         if "float_vs_int_trainer" in r]
    if c:
        print(f"C diagnostic: mean |log p_float_trainer − log p_int| = {stt.mean(c):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = load(a.runs)
    plots(R, a.out)
    tables(R)


if __name__ == "__main__":
    main()
