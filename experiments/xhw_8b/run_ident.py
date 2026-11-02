"""Cross-hardware bit identity of the exact engine: per-position SHA-256 of the int64 logits rows.

Workload (workload_T0.7_8x128.json): the first 8 samples of A11's T = 0.7 set (sid 0..7), prompt plus the
first 128 generated tokens, 595 positions in all. Teacher forcing: the inputs are the recorded tokens, so a
run on a subset (--texts K --positions N) computes a prefix of the full set's rows, bit for bit.
Per position "sid:t" (t-th generated token, logits from the row before it) the run stores
[sha256 of the logits row's int64 bytes, greedy token, T = 0.7 integer Gumbel-max token (noise key (sid, t))].

Modes: bs1   each sample alone, one forward over prompt + tokens
       bs8   8 samples per forward, right padding with PAD
       incr  prefill all prompts in one padded batch, then one token per step for all rows (KV cache)

  python experiments/xhw_8b/run_ident.py --const const_8b.pt --device cuda:0 --mode bs1 --out h800_bs1.json
  python experiments/xhw_8b/run_ident.py --const const_8b.pt --device cpu --mode bs1 --texts 2 --positions 16 \
      --cpu-lazy --out m4_bs1.json
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from bitexact.exact.model import ExactLlama  # noqa: E402
from bitexact.generate import choose  # noqa: E402
from bitexact.intops import first_argmax  # noqa: E402
from bitexact.llama import PAD  # noqa: E402

WORKLOAD = os.path.join(HERE, "workload_T0.7_8x128.json")
TN, TD = 7, 10                                    # T = 0.7
CHUNK = 128                                       # logits rows per lm-head call


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows_digest(model, logits, keys):
    """[sha256, greedy, sampled] per row of logits [M, V]."""
    greedy = first_argmax(logits).tolist()
    samp = choose(model, logits, keys, TN, TD).tolist()
    lg = logits.cpu()
    return [[hashlib.sha256(lg[r].contiguous().numpy().tobytes()).hexdigest(), greedy[r], samp[r]]
            for r in range(lg.shape[0])]


def run_full(model, seqs, B, dev, log):
    out = {}
    for b0 in range(0, len(seqs), B):
        grp = seqs[b0:b0 + B]
        Lmax = max(len(s["ids"]) for s in grp)
        ids = torch.full((len(grp), Lmax), PAD, dtype=torch.long)
        for r, s in enumerate(grp):
            ids[r, :len(s["ids"])] = torch.tensor(s["ids"])
        pos = torch.arange(Lmax).expand(len(grp), Lmax).contiguous()
        h = model.forward(ids.to(dev), pos.to(dev), model.new_cache(len(grp), Lmax))
        for r, s in enumerate(grp):
            P, n = s["start"], len(s["ids"]) - s["start"]
            for c0 in range(0, n, CHUNK):
                c1 = min(n, c0 + CHUNK)
                lg = model.logits(h[r, P - 1 + c0:P - 1 + c1])
                for t, d in enumerate(rows_digest(model, lg, [(s["sid"], t) for t in range(c0, c1)])):
                    out[f"{s['sid']}:{c0 + t}"] = d
        log(f"group {b0 // B}: {len(out)} positions")
    return out


def run_incr(model, seqs, dev, log):
    Bn = len(seqs)
    P = torch.tensor([s["start"] for s in seqs])
    n = [len(s["ids"]) - s["start"] for s in seqs]
    cache = model.new_cache(Bn, max(len(s["ids"]) for s in seqs))
    Pm = int(P.max())
    ids = torch.full((Bn, Pm), PAD, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :s["start"]] = torch.tensor(s["ids"][:s["start"]])
    h = model.forward(ids.to(dev), torch.arange(Pm).expand(Bn, Pm).contiguous().to(dev), cache)
    last = h[torch.arange(Bn), (P - 1).to(dev)]
    out = {}
    for t in range(max(n)):
        lg = model.logits(last)
        for r, d in enumerate(rows_digest(model, lg, [(s["sid"], t) for s in seqs])):
            if t < n[r]:
                out[f"{seqs[r]['sid']}:{t}"] = d
        if t + 1 >= max(n):
            break
        # rows already done feed PAD at their last position again (rewrites the same cache slot)
        tok = torch.tensor([s["ids"][s["start"] + t] if t + 1 < n[r] else PAD for r, s in enumerate(seqs)])
        pos = torch.tensor([s["start"] + t if t + 1 < n[r] else s["start"] + n[r] - 1 for r, s in enumerate(seqs)])
        last = model.forward(tok[:, None].to(dev), pos[:, None].to(dev), cache)[:, 0]
        if t % 16 == 0:
            log(f"step {t}")
    return out


def load_workload(path, texts, positions):
    w = json.load(open(path))
    seqs = sorted(w["seqs"], key=lambda s: s["sid"])[:texts]
    for s in seqs:
        s["ids"] = s["ids"][:s["start"] + positions]
    return w, seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", required=True)
    ap.add_argument("--device", default="cuda:0", help="cuda[:i] or cpu (MPS has no float64 / int64 GEMM)")
    ap.add_argument("--mode", choices=["bs1", "bs8", "incr"], required=True)
    ap.add_argument("--workload", default=WORKLOAD)
    ap.add_argument("--texts", type=int, default=8, help="first K samples (by sid)")
    ap.add_argument("--positions", type=int, default=128, help="first N generated positions per sample")
    ap.add_argument("--gemm", default="triton", choices=["triton", "cublas"])
    ap.add_argument("--compiled", action="store_true", help="torch.compile the element-wise stages (CUDA)")
    ap.add_argument("--cpu-lazy", action="store_true", help="CPU: float64 weights per call (RAM < 80 GB)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: torch default)")
    ap.add_argument("--expect-sha", default="2f02ac8e", help="prefix of the constants' internal SHA-256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device.startswith("mps"):
        sys.exit("MPS lacks float64 and int64 matrix products; use --device cpu on Apple silicon")
    if a.threads:
        torch.set_num_threads(a.threads)
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:8.1f}s] {m}", flush=True)
    w, seqs = load_workload(a.workload, a.texts, a.positions)
    c = torch.load(a.const, map_location="cpu", weights_only=False, mmap=True)
    assert c["sha256"].startswith(a.expect_sha), f"constants sha256 {c['sha256']} != {a.expect_sha}..."
    dev = torch.device(a.device)
    model = ExactLlama(c, dev, gemm=a.gemm, compiled=a.compiled, cpu_lazy=a.cpu_lazy)
    log(f"loaded constants sha256 {c['sha256'][:16]}..., dims H={model.H} HKV={model.HKV} HD={model.HD} "
        f"D={model.D} layers={model.L}")
    del c
    t0 = time.time()
    with torch.no_grad():
        if a.mode == "incr":
            rows = run_incr(model, seqs, dev, log)
        else:
            rows = run_full(model, seqs, int(a.mode[2:]), dev, log)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    model.check()                                 # raises on any integer bound violation
    env = {"torch": torch.__version__, "host": platform.node(), "machine": platform.machine(),
           "processor": platform.processor(), "python": platform.python_version(),
           "threads": torch.get_num_threads(), "gemm": model.lm.gemm, "compiled": a.compiled,
           "cpu_lazy": a.cpu_lazy}
    if dev.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(dev)
        env["cuda"] = torch.version.cuda
    try:
        import triton
        env["triton"] = triton.__version__
    except ImportError:
        pass
    res = {"schema": "xhw_8b.ident.v1", "constants_sha256": model.sha, "workload_sha256": file_sha(a.workload),
           "workload_name": w.get("name"), "mode": a.mode, "device": a.device, "texts": a.texts,
           "positions": a.positions, "sec": round(sec, 1), "sec_per_position": round(sec / max(1, len(rows)), 3),
           "n_rows": len(rows), "env": env, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(res, f)
    log(f"{a.mode} {a.device} {len(rows)} positions {sec:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
