"""Measured value ranges of the int8 engine's integer tensors on real scorer inputs (eager, nl = 2): max |x| in bits
per tensor kind over all layers, to see which int64 tensors would fit int32 (< 2^31) and how many base-256 limbs
q / k / v / p use.

  python range_probe.py --const C.pt --model TOK_DIR --data GSM_DIR --rollout R.json --bs 8 --nbatch 4 --out OUT.json
"""
import argparse
import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from prof_scorer import load_engine, score_batch  # noqa: E402

MAX = collections.defaultdict(int)


def rec(name, t):
    if torch.is_tensor(t) and t.numel():
        MAX[name] = max(MAX[name], int(t.abs().max()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--nbatch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.nl, a.compiled = 2, False
    import bitexact.int8.segs as S
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())][:a.bs * a.nbatch]
    m = load_engine(a, items)

    def wrap(name, f, ins):
        def g(*x):
            for i, nm in ins:
                rec(f"{name}.in.{nm}", x[i])
            out = f(*x)
            for j, o in enumerate(out if isinstance(out, tuple) else (out,)):
                rec(f"{name}.out{j}", o)
            return out
        return g
    S.pre_norm = wrap("pre_norm", S.pre_norm, [(0, "x_residual")])
    S.post = wrap("post", S.post, [(0, "acc_joined")])
    m.sg.qkv = wrap("qkv", m.sg.qkv, [(0, "y")])                 # bound directly in Segs (no global lookup)
    m.sg.softmax = wrap("softmax", m.sg.softmax, [(0, "raw_qk")])
    S.attn_out = wrap("attn_out", S.attn_out, [(0, "num_pv"), (1, "den")])
    S.silu_mul = wrap("silu_mul", S.silu_mul, [(0, "gate_up")])
    rsh0 = S.rsh

    def rsh(x, s):                                   # every rounding shift input: the largest int64 intermediate
        rec("rsh.in(any)", x)
        return rsh0(x, s)
    S.rsh = rsh
    gemm = m._gemm
    m._gemm = lambda x, w: (lambda c: (rec("int_mm.out_int32", c), c)[1])(gemm(x, w))
    lg = m.logits
    m.logits = lambda x: (lambda y: (rec("logits", y), y)[1])(lg(x))
    for i in range(0, len(items), a.bs):
        score_batch(m, items[i:i + a.bs], a.chunk)
    names = {"pre_norm.out0": "q of RMSNorm input (limbs, nl=2)", "qkv.out0": "q after RoPE (FQ 14)",
             "qkv.out1": "k after RoPE (FQ 14)", "qkv.out2": "v (FV 14)", "softmax.out0": "p (FP 20)",
             "softmax.out1": "den = sum p"}
    out = {k: {"max": v, "bits": v.bit_length(), "limbs": (v.bit_length() + 1 + 7) // 8, "fits_int32": v < 2 ** 31,
               "what": names.get(k, "")} for k, v in sorted(MAX.items())}
    for k, v in out.items():
        print(f"{k:28s} bits {v['bits']:2d} limbs {v['limbs']} int32 {v['fits_int32']}  {v['what']}")
    json.dump({"seqs": len(items), "rollout": os.path.basename(a.rollout), "ranges": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
