"""Markdown tables for RESULTS.md from results/*.json.   python gen_tables.py"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ld = lambda f: json.load(open(os.path.join(R, f)))
q1, q2 = ld("metrics_q1.json")["mismatch"], ld("metrics_q2.json")
q1["D_vllm"] = ld("metrics_q1_D.json")["mismatch"]["D_vllm"]
q2.update(ld("metrics_q2_D.json"))
f = lambda x, p=3: f"{x:.{p}g}"
pc = lambda x: f"{100 * x:.2f}%" if x else "0"

print("### Q1\n\n| arm (sampler -> scorer) | tokens | mean abs d | max abs d | tokens d!=0 | token r outside clip | argmax differs | sequence r outside clip |\n|---|---|---|---|---|---|---|---|")
for k, name in [("A_orig_hfbf16", "A_orig: vLLM bf16 (orig) -> HF bf16"), ("A_orig_hffp32", "A_orig: vLLM bf16 (orig) -> HF fp32"),
                ("A_hfbf16", "A: vLLM bf16 -> HF bf16"), ("A_hffp32", "A: vLLM bf16 -> HF fp32"),
                ("B_hfbf16", "B: plugin -> HF bf16"), ("B_hffp32", "B: plugin -> HF fp32"), ("C_engine", "C: plugin -> BitExact engine"), ("D_vllm", "D: vLLM batch-invariant -> same engine (prompt_logprobs)")]:
    v = q1[k]
    print(f"| {name} | {v['tokens']:,} | {f(v['absd_mean'])} | {f(v['absd_max'])} | {pc(v['frac_tokens_d_nonzero'])} | {pc(v['frac_tokens_outside_clip'])} | {pc(v['frac_argmax_differs'])} | {pc(v['frac_seqs_ratio_outside_clip'])} |")

print("\n### Q2 per-position bucket, mean abs d (tokens in bucket)\n")
names = [("A_hfbf16", "A, HF bf16"), ("A_hffp32", "A, HF fp32"), ("B_hfbf16", "B, HF bf16"), ("B_hffp32", "B, HF fp32"), ("C_engine", "C"), ("D_vllm", "D")]
print("| run | " + " | ".join(["0-256", "256-1k", "1k-4k", "4k-16k"]) + " |\n|---|---|---|---|---|")
for L in (256, 1024, 4096, 16384):
    for k, nm in names:
        r = q2[f"{k}_{L}"]
        cells = {tuple(b["range"]): b for b in r["buckets"]}
        row = []
        for rg in [(0, 256), (256, 1024), (1024, 4096), (4096, 16384)]:
            b = cells.get(rg)
            row.append(f"{f(b['mean_abs_d'])}" if b else "")
        print(f"| {L}-token gens ({r['seqs']} seqs), {nm} | " + " | ".join(row) + " |")

print("\n### Q2 cumulative sequence log-ratio S_T (sum of d over first T tokens): mean abs S / mean S / fraction of sequences with exp(S) outside [0.8,1.2]\n")
print("| gen length (seqs) | arm | T=256 | T=1k | T=4k | T=16k |\n|---|---|---|---|---|---|")
for L in (256, 1024, 4096, 16384):
    for k, nm in names:
        r = q2[f"{k}_{L}"]
        cells = {c["T"]: c for c in r["checkpoints"]}
        row = []
        for T in (256, 1024, 4096, 16384):
            c = cells.get(T)
            row.append(f"{f(c['mean_abs_S'])} / {f(c['mean_S'])} / {pc(c['frac_seq_r_outside_clip'])}" if c else "")
        print(f"| {L} ({r['seqs']}) | {nm} | " + " | ".join(row) + " |")

print("\n### Q4 sampler tok/s (median of 3)\n\n| gen len | batch | vLLM bf16 (A) | BitExact plugin (B, C) | vLLM batch-invariant (D) | A / plugin | A / D |\n|---|---|---|---|---|---|---|")
a = {(r["gen_len"], r["batch"]): r for r in ld("speed_samp_bf16.json")["rows"]}
b = {(r["gen_len"], r["batch"]): r for r in ld("speed_samp_bx.json")["rows"]}
d = {(r["gen_len"], r["batch"]): r for r in ld("speed_samp_D.json")["rows"]}
for k in sorted(a):
    print(f"| {k[0]} | {k[1]} | {a[k]['tok_per_s_median']:,.0f} | {b[k]['tok_per_s_median']:,.0f} | {d[k]['tok_per_s_median']:,.0f} | {a[k]['tok_per_s_median'] / b[k]['tok_per_s_median']:.1f}x | {a[k]['tok_per_s_median'] / d[k]['tok_per_s_median']:.1f}x |")
print("\n### Q4 scorer tok/s (median of 3; tokens = prompt + response tokens pushed through the network)\n\n| workload | HF bf16 | HF fp32 | BitExact engine (C) | vLLM batch-invariant, prompt_logprobs (D) |\n|---|---|---|---|---|")
for w, pre in [("1024 GSM8K rollouts (215,899 response tokens), batch 8", "gsm"), ("16 x 4096-token sequences, batch 4", "4k")]:
    v = [ld(f"speed_sc_{pre}_{m}.json")["rows"][0] for m in ("hf_bf16", "hf_fp32", "engine")]
    dd = {"gsm": ["speed_sc_gsm_D8.json", "speed_sc_gsm_D64.json"], "4k": ["speed_sc_4k_D4.json"]}[pre]
    dv = " / ".join(f"{ld(x)['meta']['tok_per_s_median']:,.0f} (max_num_seqs {ld(x)['meta']['max_seqs']})" for x in dd)
    print(f"| {w} | " + " | ".join(f"{x['tok_per_s_median']:,.0f}" for x in v) + f" | {dv} |")
