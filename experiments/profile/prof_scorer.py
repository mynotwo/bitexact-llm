"""Profile the standalone int8 engine as a scorer (teacher-forced log p), as in abc_measure/speed_scorer.py.

  python prof_scorer.py --const C.pt --model TOK_DIR --data GSM_DIR --rollout R.json --bs 8 --out OUT_PREFIX
        [--compiled] [--nl 2] [--prof-batches 4] [--time-batches 0] [--chunk 256] [--bytes]

Scoring is the exact speed_scorer.py engine path: right-padded batch, prefill in --chunk position chunks with
S = keys up to the chunk end, then lm_head + float64 log-softmax per response chunk, one .item() each.
Writes OUT_PREFIX.json with
  timing   : wall time over --time-batches batches (0 = all), unprofiled, after one warm-up batch
  padding  : real vs padded tokens, real vs computed query-key pairs
  profile  : torch.profiler over the first --prof-batches batches (eager: category ranges; compiled: kernel names)
  gemm     : per (M, N, K) int8 GEMM shape: count, kernel time, achieved TOPS
  bytes    : (--bytes) logical bytes per category for one batch (dispatch-mode counter)
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
import instrument  # noqa: E402
from bitexact_engine import _pad  # noqa: E402


def load_engine(a, items):
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin.model import _extend_rope
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model, max(len(p) + len(r) for _, p, r in items) + 8)
    return IntLlama(c, "cuda", nl=a.nl, compiled=a.compiled)


def score_batch(m, b, chunk):
    ids, pos, L = _pad(m, [p + r for _, p, r in b])
    cache = m.new_cache(len(b), L)
    with instrument.cat("prefill"):
        hs = [m.forward(ids[:, c:min(L, c + chunk)], pos[:, c:min(L, c + chunk)], cache, S=min(L, c + chunk))
              for c in range(0, L, chunk)]
        h = torch.cat(hs, 1)
    for j, (_, p, r) in enumerate(b):
        P, n = len(p), len(r)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            lg = m.logits(h[j, P - 1 + c0:P - 1 + c1])
            with instrument.cat("logp"):
                lg = lg.double() * 2.0 ** -16
                lg.log_softmax(-1)[torch.arange(c1 - c0), torch.tensor(r[c0:c1], device=lg.device)].sum().item()
    m.check()


def padding_stats(batches, chunk):
    real = comp = 0
    pairs_real = pairs_comp = 0
    for b in batches:
        Ls = [len(p) + len(r) for _, p, r in b]
        L = max(Ls)
        real += sum(Ls)
        comp += len(b) * L
        pairs_real += sum(x * (x + 1) // 2 for x in Ls)
        for c in range(0, L, chunk):
            T = min(L, c + chunk) - c
            pairs_comp += len(b) * T * min(L, c + chunk)          # every query of the chunk x S keys
    return {"real_tokens": real, "computed_tokens": comp, "token_util": round(real / comp, 4),
            "real_qk_pairs": pairs_real, "computed_qk_pairs": pairs_comp, "qk_util": round(pairs_real / pairs_comp, 4)}


def gemm_table(kernels):
    """aten::_int_mm kernels: shape (M, K) x (K, N) from the op's input dims."""
    agg = collections.defaultdict(lambda: [0, 0.0])
    for name, dur, path, op, dims, *_ in kernels:
        if op == "aten::_int_mm" and dims and len(dims) >= 2:
            (M, K), (_, N) = dims[0], dims[1]
            agg[(M, N, K, "lm_head" in path)][0] += 1
            agg[(M, N, K, "lm_head" in path)][1] += dur
    rows = []
    for (M, N, K, lm), (cnt, us) in sorted(agg.items(), key=lambda x: -x[1][1]):
        tops = 2 * M * N * K * cnt / (us * 1e-6) / 1e12
        rows.append({"M": M, "N": N, "K": K, "lm_head": lm, "count": cnt, "us": round(us, 1),
                     "us_each": round(us / cnt, 2), "TOPS": round(tops, 1)})
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--prof-batches", type=int, default=4)
    ap.add_argument("--time-batches", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--bytes", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
    batches = [items[i:i + a.bs] for i in range(0, len(items), a.bs)]
    m = load_engine(a, items)
    res = {"args": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "padding": padding_stats(batches, a.chunk)}

    score_batch(m, batches[0], a.chunk)                      # warm-up (and compile)
    tb = batches if a.time_batches == 0 else batches[:a.time_batches]
    ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize()
        t0 = time.time()
        for b in tb:
            score_batch(m, b, a.chunk)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    tin = sum(len(p) + len(r) for b in tb for _, p, r in b)
    res["timing"] = {"batches": len(tb), "seq_tokens": tin, "sec": [round(x, 3) for x in ts],
                     "tok_per_s": round(tin / statistics.median(ts), 1)}
    print(json.dumps(res["timing"]), flush=True)

    pb = batches[:a.prof_batches]
    instrument.annotate(m, fine=not a.compiled)
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        t0 = time.time()
        for b in pb:
            score_batch(m, b, a.chunk)
        torch.cuda.synchronize()
        wall = time.time() - t0
    trace = a.out + ".trace.json"
    prof.export_chrome_trace(trace)
    at = instrument.attribute(trace)
    ptok = sum(len(p) + len(r) for b in pb for _, p, r in b)
    by_class = collections.defaultdict(lambda: [0.0, 0])
    for name, dur, path, op, dims, _ in at["kernels"]:
        k = instrument.kernel_class(name)
        by_class[k][0] += dur
        by_class[k][1] += 1
    res["profile"] = {"batches": len(pb), "seq_tokens": ptok, "wall_s_profiled": round(wall, 3),
                      "busy_us": round(at["busy_us"]), "span_us": round(at["span_us"]), "n_kernels": at["n_kernels"],
                      "by_path": {k: [round(v[0], 1), v[1]] for k, v in sorted(at["by_path"].items(), key=lambda x: -x[1][0])},
                      "by_kernel_class": {k: [round(v[0], 1), v[1]] for k, v in sorted(by_class.items(), key=lambda x: -x[1][0])}}
    top = collections.defaultdict(lambda: [0.0, 0])
    for name, dur, *_ in at["kernels"]:
        top[name[:120]][0] += dur
        top[name[:120]][1] += 1
    res["top_kernels"] = [[k, round(v[0], 1), v[1]] for k, v in sorted(top.items(), key=lambda x: -x[1][0])[:40]]
    res["gemm"] = gemm_table(at["kernels"])
    os.remove(trace) if os.path.getsize(trace) > 3e8 else None

    if a.bytes and not a.compiled:
        bc = instrument.ByteCounter()
        m2 = m
        import bitexact.int8.model as M
        inner = M.bmm.__wrapped__

        def bmm(x, y, *r, **k):            # the Triton kernel is not an aten op: count inputs + output once
            c = inner(x, y, *r, **k)
            bc.add("/".join(instrument.STACK), x.numel() * x.element_size() + y.numel() * y.element_size()
                   + c.numel() * c.element_size())
            return c
        M.bmm = lambda x, y, *r, **k: _bmm_named(x, y, bmm, *r, **k)
        _bmm_named.n = 0
        with bc:
            score_batch(m2, batches[0], a.chunk)
        res["bytes"] = {"batch_tokens_real": sum(len(p) + len(r) for _, p, r in batches[0]),
                        "batch_tokens_padded": len(batches[0]) * max(len(p) + len(r) for _, p, r in batches[0]),
                        "by_path": dict(bc.bytes.most_common()), "ops": dict(bc.ops.most_common())}
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print("WROTE", a.out + ".json", flush=True)


def _bmm_named(x, y, f, *r, **k):
    name = "attn_qk" if _bmm_named.n % 2 == 0 else "attn_pv"
    _bmm_named.n += 1
    with instrument.cat(name):
        return f(x, y, *r, **k)


if __name__ == "__main__":
    main()
