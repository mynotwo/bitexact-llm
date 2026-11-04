"""Scorer throughput on one GPU: HF bf16 / HF fp32 / BitExact int8 engine, teacher-forced log p.

  python speed_scorer.py --mode hf_bf16|hf_fp32|engine --model HF_DIR --data GSM_DIR --out S.json [--const C.pt]

Workloads (sequences of prompt + response tokens, random-free: real GSM8K prompt followed by tokens taken from a
rollout file for 'gsm', or the same prompt tokens repeated to fixed length for 'long'):
  gsm  : the 1024 rollouts of the main 256-token run, batch 8, right padded
  long : 8 sequences of 4096 tokens (response = 4096 - prompt), batch 8 -> here taken from the 4k rollout file
Per rep: wall time of forward + log p of the response positions (log-softmax and gather on the GPU, float32 for HF,
float64 for the engine's integer logits), no host transfer of rows. tok/s = scored sequence tokens
(prompt + response, i.e. tokens pushed through the network) / time; resp tok/s counts response tokens only.
Warm-up: one untimed pass over the first batch. 3 reps, median.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from bitexact_engine import _pad  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf_bf16", "hf_fp32", "engine"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--const")
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", nargs="+", required=True, help="NAME=ROLLOUT.json[:BS] workloads")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    if a.mode == "engine":
        from bitexact import load
        m = load(a.const, "cuda")
    else:
        from transformers import AutoModelForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        dt = torch.bfloat16 if a.mode == "hf_bf16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").cuda().eval()

    def score_batch(b):
        if a.mode == "engine":
            ids, pos, L = _pad(m, [p + r for _, p, r in b])
            cache = m.new_cache(len(b), L)
            hs = [m.forward(ids[:, c:min(L, c + a.chunk)], pos[:, c:min(L, c + a.chunk)], cache, S=min(L, c + a.chunk))
                  for c in range(0, L, a.chunk)]
            h = torch.cat(hs, 1)
            for j, (_, p, r) in enumerate(b):
                P, n = len(p), len(r)
                for c0 in range(0, n, a.chunk):
                    c1 = min(n, c0 + a.chunk)
                    lg = m.logits(h[j, P - 1 + c0:P - 1 + c1]).double() * 2.0 ** -16
                    lg.log_softmax(-1)[torch.arange(c1 - c0), torch.tensor(r[c0:c1], device=lg.device)].sum().item()
            m.check()
        else:
            L = max(len(p) + len(r) for _, p, r in b)
            ids = torch.zeros((len(b), L), dtype=torch.long)
            mask = torch.zeros((len(b), L), dtype=torch.long)
            for j, (_, p, r) in enumerate(b):
                ids[j, :len(p) + len(r)] = torch.tensor(p + r)
                mask[j, :len(p) + len(r)] = 1
            h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda() if len(b) > 1 else None).last_hidden_state
            for j, (_, p, r) in enumerate(b):
                P, n = len(p), len(r)
                for c0 in range(0, n, a.chunk):
                    c1 = min(n, c0 + a.chunk)
                    lg = model.lm_head(h[j, P - 1 + c0:P - 1 + c1]).float().log_softmax(-1)
                    lg[torch.arange(c1 - c0), torch.tensor(r[c0:c1], device=lg.device)].sum().item()
        torch.cuda.synchronize()

    rows = []
    for spec in a.work:
        name, rest = spec.split("=")
        path, _, bs = rest.partition(":")
        bs = int(bs or 8)
        rs = common.load(path)["seqs"]
        items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
        batches = [items[i:i + bs] for i in range(0, len(items), bs)]
        score_batch(batches[0])
        t_in = sum(len(p) + len(r) for _, p, r in items)
        t_resp = sum(len(r) for _, _, r in items)
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.time()
            for b in batches:
                score_batch(b)
            ts.append(time.time() - t0)
        row = {"mode": a.mode, "workload": name, "batch": bs, "seqs": len(items), "seq_tokens": t_in,
               "resp_tokens": t_resp, "sec_reps": [round(x, 2) for x in ts],
               "tok_per_s_median": round(t_in / statistics.median(ts), 1),
               "resp_tok_per_s_median": round(t_resp / statistics.median(ts), 1)}
        print(json.dumps(row), flush=True)
        rows.append(row)
        json.dump({"gpu": torch.cuda.get_device_name(0), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
