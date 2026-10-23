"""Record logits hashes on one device, then compare records from several devices / configurations.

Each run prefills fixed prompts and generates --max-new tokens per prompt through the KV cache with
Gumbel-max sampling at T = 0.7 (noise keyed (prompt index, t)), so any differing bit also changes the
continuation. It stores the SHA-256 of every generated position's int64 logits row and the token.

  python examples/check_bit_identity.py record --const C --device cuda:0 --out gpu0.json
  python examples/check_bit_identity.py record --const C --device cpu --out cpu.json
  python examples/check_bit_identity.py compare gpu0.json cpu.json ...
"""

import argparse
import json
import platform
import sys

import torch

from bitexact import generate, load

PROMPTS = [  # Llama-3 token ids (BOS + a short English sentence; any ids work)
    [128000, 791, 23070, 19044, 6267, 304, 813, 1515, 430, 279, 13766, 1047],
    [128000, 755, 11117, 91907, 1577, 43298, 997, 262, 4304, 52586, 50917, 28090, 14781, 198],
    [128000, 644, 220, 9674, 17, 279, 33715, 8813, 279, 18671, 13962, 11, 1405, 279, 6098, 15916],
    [128000, 29880, 369, 264, 4062, 42120, 19724, 25, 1176, 11, 8798, 1403, 56488, 315, 33213, 5707, 323],
]


def record(a):
    torch.set_num_threads(a.threads)
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    kw = {"compiled": a.compiled, **({"gemm": a.gemm} if "lm.limbs" in c else {})}
    m = load(c, a.device, **kw)
    rows = {}
    for i, ids in enumerate(PROMPTS):
        toks, hashes = generate(m, ids, a.max_new, 7, 10, seed=i)
        rows.update({f"{i}:{t}": [h, y] for t, (h, y) in enumerate(zip(hashes, toks))})
    env = {"torch": torch.__version__, "device": a.device, "kw": kw, "model_sha256": c["sha256"]}
    if a.device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(torch.device(a.device))
    else:
        env["cpu"] = platform.processor() or platform.machine()
    with open(a.out, "w") as f:
        json.dump({"env": env, "rows": rows}, f)
    print(a.out, len(rows), "positions")


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    ref = runs[0]["rows"]
    bad = False
    for p, r in zip(paths, runs):
        common = set(ref) & set(r["rows"])
        dl = sum(ref[k][0] != r["rows"][k][0] for k in common)
        dt = sum(ref[k][1] != r["rows"][k][1] for k in common)
        print(f"{p}: positions {len(common)} logits_differ {dl} tokens_differ {dt} env {json.dumps(r['env'])}")
        bad |= bool(dl or dt or len(common) != len(ref))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--const", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--gemm", default="triton")
    r.add_argument("--compiled", action="store_true")
    r.add_argument("--max-new", type=int, default=128)
    r.add_argument("--threads", type=int, default=torch.get_num_threads())
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a)
        return 0
    return compare(a.paths)


if __name__ == "__main__":
    sys.exit(main())
