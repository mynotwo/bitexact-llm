"""Compare runs (vLLM): tokens (OUT.json) and per-position logits hashes (OUT.hash).

  python tests/vllm/compare.py REF RUN [RUN ...]     (paths without extension)

Only the reference's target seeds are compared (fillers use seeds >= 1000, vLLM's dummy warm-up and
capture runs log seed 0). A hash file can hold repeated (seed, t) rows (vLLM recomputes
after preemption); all copies must agree. Prints one line per run and exits 1 on any difference.
"""

import json
import sys


def load(p):
    toks = {int(k): v for k, v in json.load(open(p + ".json")).items()}
    hs, dup_bad = {}, 0
    for ln in open(p + ".hash"):
        s, t, h, y = ln.split()
        s, t = int(s), int(t)
        if s >= 1000:
            continue
        if (s, t) in hs and hs[(s, t)] != (h, int(y)):
            dup_bad += 1
        hs[(s, t)] = (h, int(y))
    return toks, hs, dup_bad


def main():
    ref, rh, _ = load(sys.argv[1])
    bad = False
    for p in sys.argv[2:]:
        toks, hs, dup_bad = load(p)
        hs = {k: v for k, v in hs.items() if k[0] in ref}        # vLLM's dummy warm-up rows use seed 0
        tok_diff = sum(toks.get(s) != v for s, v in ref.items())
        keys = set(rh) | set(hs)
        miss = sum(k not in hs or k not in rh for k in keys)
        h_diff = sum(rh[k][0] != hs[k][0] for k in keys if k in hs and k in rh)
        print(f"{p}: targets {len(ref)} token_seqs_diff {tok_diff} positions {len(keys)} missing {miss} "
              f"logit_hash_diff {h_diff} inconsistent_repeats {dup_bad}")
        bad |= bool(tok_diff or miss or h_diff or dup_bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
