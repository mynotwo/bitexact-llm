"""Does the scorer with compiled=True (the engine's existing torch.compile option for its elementwise segments) give
the same integer logits as the plugin sampler? Compares the sha16 of every response logits row with the sha16 the
plugin recorded in the rollout file (arm C of abc_measure: the eager scorer matched all 215,899 rows).

  python check_compiled.py --const C.pt --model TOK_DIR --data GSM_DIR --rollout R.json --bs 8 --nseq 64 [--compiled]
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from bitexact_engine import _pad  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--nseq", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--compiled", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin.model import _extend_rope
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"], v["sha"]) for s, v in sorted(rs.items())][:a.nseq]
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model, max(len(p) + len(r) for _, p, r, _ in items) + 8)
    m = IntLlama(c, "cuda", nl=2, compiled=a.compiled)
    rows = same = 0
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]
        ids, pos, L = _pad(m, [p + r for _, p, r, _ in b])
        cache = m.new_cache(len(b), L)
        h = torch.cat([m.forward(ids[:, c0:min(L, c0 + a.chunk)], pos[:, c0:min(L, c0 + a.chunk)], cache,
                                 S=min(L, c0 + a.chunk)) for c0 in range(0, L, a.chunk)], 1)
        for j, (_, p, r, sha) in enumerate(b):
            lg = m.logits(h[j, len(p) - 1:len(p) - 1 + len(r)]).cpu().numpy()
            for t in range(len(r)):
                rows += 1
                same += hashlib.sha256(lg[t].tobytes()).hexdigest()[:16] == sha[t]
    m.check()
    out = {"compiled": a.compiled, "rollout": os.path.basename(a.rollout), "seqs": len(items), "rows": rows,
           "rows_equal_to_plugin_sha": same}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
