"""BitExact standalone int8 engine (no vLLM): trainer-side rescoring and batched replay generation.

  python bitexact_engine.py score    --const C --rollout B.json --tokenizer DIR --data GSM_DIR --device cuda --bs 8 --out S.json
  python bitexact_engine.py generate --const C --rollout B.json --tokenizer DIR --data GSM_DIR --device cuda --bs 64 --out G.json

score: teacher forcing on prompt + the sampled tokens, one full forward per right-padded batch.
generate: re-sample the same (prompt, seed) requests: right-padded prefill of a batch, then one token per
step for every row through the integer KV cache, Gumbel-max at T = 1 keyed (seed, t) as in the plugin.
Both record per position the sha16 of the int64 logits row, its first argmax and common.int_logp.
--seeds-from FILE (one seed per line) limits the run to those requests (e.g. a CPU subset).
"""

import argparse
import hashlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

EOS = 128001


def _rows(lg, toks):
    """int64 logits [n, V], tokens [n] -> (sha16, argmax, logp) lists."""
    from bitexact.intops import first_argmax
    am = first_argmax(lg).tolist()
    rows = lg.cpu().numpy()
    sha = [hashlib.sha256(rows[i].tobytes()).hexdigest()[:16] for i in range(len(toks))]
    return sha, am, [common.int_logp(rows[i], toks[i]) for i in range(len(toks))]


def _pad(m, seqs):
    from bitexact.llama import PAD
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    pos = torch.arange(L).expand(len(seqs), L).contiguous()
    return ids.to(m.dev), pos.to(m.dev), L


@torch.no_grad()
def score(m, items, bs):
    out = {}
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        ids, pos, L = _pad(m, [p + r for _, p, r in b])
        h = m.forward(ids, pos, m.new_cache(len(b), L))
        for j, (s, p, r) in enumerate(b):
            P, n = len(p), len(r)
            sha, am, lp = _rows(m.logits(h[j, P - 1:P - 1 + n]), r)
            out[s] = {"sha": sha, "am": am, "lp": lp}
    m.check()
    return out


@torch.no_grad()
def generate(m, items, bs, max_new):
    from bitexact.generate import choose
    out = {}
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        B = len(b)
        Ps = torch.tensor([len(p) for _, p, _ in b], device=m.dev)
        ids, pos, L = _pad(m, [p for _, p, _ in b])
        cache = m.new_cache(B, L + max_new)
        h = m.forward(ids, pos, cache)[torch.arange(B, device=m.dev), Ps - 1]
        rec = [{"tok": [], "sha": [], "am": [], "lp": []} for _ in b]
        live = [True] * B
        for t in range(max_new):
            lg = m.logits(h)
            y = choose(m, lg, [(s, t) for s, _, _ in b], 1, 1)
            yl = y.tolist()
            sha, am, lp = _rows(lg, yl)
            for j in range(B):
                if live[j]:
                    for k, v in zip(("tok", "sha", "am", "lp"), (yl[j], sha[j], am[j], lp[j])):
                        rec[j][k].append(v)
                    live[j] = yl[j] != EOS
            if not any(live) or t + 1 == max_new:
                break
            h = m.forward(y.view(B, 1), (Ps + t).view(B, 1), cache)[:, 0]
        for (s, _, _), r in zip(b, rec):
            out[s] = r
    m.check()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "generate"])
    ap.add_argument("--const", required=True)
    ap.add_argument("--rollout", required=True, help="rollout JSON: tokens to score / requests to replay")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--order", choices=["group", "sorted"], default="group")
    ap.add_argument("--seeds-from")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer

    from bitexact import load
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    keep = set(int(x) for x in open(a.seeds_from).read().split()) if a.seeds_from else None
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items()) if keep is None or s in keep]
    if a.order == "sorted":
        items.sort(key=lambda x: len(x[1]) + len(x[2]))
    m = load(a.const, a.device)
    t0 = time.time()
    if a.cmd == "score":
        out = score(m, items, a.bs)
    else:
        out = generate(m, items, a.bs, common.MAX_NEW)
        for s, r in out.items():
            r["p"] = rs[s]["p"]
            r["text"] = tok.decode([x for x in r["tok"] if x != EOS])
            r["reward"], r["flag"] = common.grade(r["text"], prompts[r["p"]][1])
    dev = torch.cuda.get_device_name(m.dev) if m.dev.type == "cuda" else f"cpu x{torch.get_num_threads()}"
    meta = dict(vars(a), sec=round(time.time() - t0, 1), device_name=dev, torch=torch.__version__,
                const_sha=m.sha, n_seqs=len(out))
    common.save(a.out, f"bitexact_{a.cmd}", meta, out)
    print(f"RESULT {a.out} seqs {len(out)} sec {meta['sec']} device {dev}")


if __name__ == "__main__":
    main()
