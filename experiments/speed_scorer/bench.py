"""Scorer speed and bit-exactness harness (teacher-forced log p over prompt + response).

  python bench.py --mode old --compiled 0 --rollout R.json --bs 8 --reps 3 --check --save-ref REF.pt   # eager reference
  python bench.py --mode packed --rollout R.json --bs 8 --reps 3 --check --ref REF.pt --order shuffle

Modes: old    = abc_measure's flow: right-padded batch, prefill in --chunk position chunks, lm_head + float64 log-softmax
                per 256-row response chunk, a host sync per chunk (IntLlama flags --compiled / --fused select the engine)
       packed = bitexact.int8.scorer.Scorer (flags --graphs, --i32 ...)
--check compares, per response row, sha256(int64 logits row)[:16] with the hash the plugin recorded in the rollout file
(and with --ref: the saved hashes and the log p array of the eager reference, bit for bit). Timing is done without
hashing (hashing copies every logits row to the host).
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from bitexact_engine import _pad  # noqa: E402


def sha_rows(lg):
    a = lg.long().cpu().numpy()
    return [hashlib.sha256(a[t].tobytes()).hexdigest()[:16] for t in range(a.shape[0])]


def old_flow(m, b, chunk=256, want_hash=False):
    """The flow of prof_scorer.score_batch; returns [(logp float64 [n], hashes or None)] per sequence."""
    ids, pos, L = _pad(m, [p + r for p, r in b])
    cache = m.new_cache(len(b), L)
    h = torch.cat([m.forward(ids[:, c:min(L, c + chunk)], pos[:, c:min(L, c + chunk)], cache, S=min(L, c + chunk))
                   for c in range(0, L, chunk)], 1)
    out = []
    for j, (p, r) in enumerate(b):
        P, n = len(p), len(r)
        lps, hs = [], []
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            lg = m.logits(h[j, P - 1 + c0:P - 1 + c1])
            if want_hash:
                hs += sha_rows(lg)
            lp = (lg.double() * 2.0 ** -16).log_softmax(-1)[torch.arange(c1 - c0), torch.tensor(r[c0:c1], device=lg.device)]
            lp.sum().item()                                   # the per-chunk host sync of the original scorer
            lps.append(lp)
        out.append((torch.cat(lps).cpu().numpy(), hs if want_hash else None))
    m.check()
    return out


def make_scorer(a, m):
    if a.mode == "old":
        return lambda b, want_hash: old_flow(m, b, a.chunk, want_hash)
    from bitexact.int8.scorer import Scorer
    kw = {k: v for k, v in (("rows", a.rows), ("sync_chunks", bool(a.sync_chunks)), ("graphs", bool(a.graphs)))
          if v is not None}
    s = Scorer(m, **kw)
    if a.warm:
        s.warmup()
    return lambda b, want_hash: s.score(b, want_hash)


def run(score, items, bs, order, want_hash):
    idx = list(range(len(items)))
    if order == "shuffle":
        random.Random(7).shuffle(idx)
    res = {}
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        for k, o in zip(sel, score([(items[k][1], items[k][2]) for k in sel], want_hash)):
            res[items[k][0]] = o
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--mode", default="old")
    ap.add_argument("--compiled", type=int, default=1)
    ap.add_argument("--fused", type=int, default=None)
    ap.add_argument("--i32", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--nseq", type=int, default=0, help="first n sequences only (0 = all)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--sync-chunks", type=int, default=None)
    ap.add_argument("--graphs", type=int, default=None)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=0, help="timed passes (0 = no timing)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--order", default="sorted", choices=["sorted", "shuffle"])
    ap.add_argument("--save-ref")
    ap.add_argument("--ref")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin.model import _extend_rope
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"], v["sha"]) for s, v in sorted(rs.items())]
    if a.nseq:
        items = items[:a.nseq]
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model, max(len(p) + len(r) for _, p, r, _ in items) + 8)
    kw = {}
    if a.fused is not None and a.device == "cuda":
        kw["fused"] = bool(a.fused)
    if a.i32:
        kw["i32"] = True
    m = IntLlama(c, a.device, nl=2, compiled=bool(a.compiled), **kw)
    score = make_scorer(a, m)
    out = {"tag": a.tag, "mode": a.mode, "compiled": a.compiled, "fused": a.fused, "bs": a.bs, "order": a.order,
           "rollout": os.path.basename(a.rollout), "seqs": len(items), "gpu": torch.cuda.get_device_name(0)
           if a.device == "cuda" else "cpu", "torch": torch.__version__}
    ok = True
    if a.check:
        res = run(score, items, a.bs, a.order, True)
        rows = same = 0
        for s, p, r, sha in items:
            hs = res[s][1]
            rows += len(sha)
            same += sum(x == y for x, y in zip(hs, sha)) if len(hs) == len(sha) else 0
        out.update(rows=rows, rows_equal_plugin_sha=same)
        ok = same == rows
        if a.save_ref:
            torch.save({s: res[s] for s in res}, a.save_ref)
        if a.ref:
            ref = torch.load(a.ref, weights_only=False)
            hs_eq = lp_eq = 0
            for s, *_ in items:
                hs_eq += res[s][1] == ref[s][1]
                lp_eq += np.array_equal(res[s][0].view(np.int64), ref[s][0].view(np.int64))
            out.update(seqs_hash_equal_ref=hs_eq, seqs_logp_bits_equal_ref=lp_eq)
            ok = ok and hs_eq == len(items) and lp_eq == len(items)
        out["bit_exact"] = bool(ok)
    if a.reps:
        run(score, items[:max(a.bs, 8)], a.bs, "sorted", False)           # warm-up
        ts = []
        ntok = sum(len(p) + len(r) for _, p, r, _ in items)
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.time()
            run(score, items, a.bs, "sorted", False)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        out.update(tokens=ntok, sec=[round(x, 3) for x in ts], tok_per_s=round(ntok / statistics.median(ts), 1))
    print(json.dumps(out), flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
