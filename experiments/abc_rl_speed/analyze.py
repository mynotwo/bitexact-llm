"""Median s/step (steps 5-30) per arm and the bit check C-new vs C-old.
python analyze.py RUNS_DIR   (RUNS_DIR holds C_new_s1, C_old_s1, D_old_s1 with steps.jsonl)"""
import json
import statistics as st
import sys

R = sys.argv[1]


def load(t):
    return {r["step"]: r for r in map(json.loads, open(f"{R}/{t}_s1/steps.jsonl")) if "time" in r}


def med(rs, f):
    return st.median(f(r) for s, r in rs.items() if 5 <= s <= 30)


print("| arm | steps | s/step | sample | score | train (fwd+bwd) | optim+quantize |")
print("|---|---|---|---|---|---|---|")
for t, name in (("C_new", "C-new (integer, packed scorer + plugin)"), ("C_old", "C-old (integer, abc_rl code)"),
                ("D_old", "D (vLLM batch-invariant)")):
    rs = load(t)
    n = len([s for s in rs if 5 <= s <= 30])
    print(f"| {name} | {n} | {med(rs, lambda r: r['step_sec']):.1f} | {med(rs, lambda r: r['time']['sample']):.1f} | "
          f"{med(rs, lambda r: r['time']['score_int']):.1f} | {med(rs, lambda r: r['time']['backward']):.1f} | "
          f"{med(rs, lambda r: r['time']['optim'] + r['time']['quantize']):.1f} |")


def bitcheck(na, nb, label):
    a, b = load(na), load(nb)
    print(f"\n{label}: {na} vs {nb}")
    for s in range(0, 10):
        if s in a and s in b:
            x, y = a[s], b[s]
            print(f"step {s}: reward {x['reward'] == y['reward']} tokens {x['tok_sha'] == y['tok_sha']} sampler_lp {x['slp_sha'] == y['slp_sha']} "
                  f"scorer_lp {x['ilp_sha'] == y['ilp_sha']} delta_max {x['delta']['max_abs']}/{y['delta']['max_abs']} "
                  f"frac_nonzero {x['delta']['frac_nonzero']}/{y['delta']['frac_nonzero']} loss_bitwise {x['loss_hex'] == y['loss_hex']} "
                  f"grad_norm_bitwise {x['grad_norm'] == y['grad_norm']}")


for args in sys.argv[2:]:
    na, nb, label = args.split(",", 2)
    bitcheck(na, nb, label)
