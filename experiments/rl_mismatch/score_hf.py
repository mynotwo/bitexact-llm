"""Trainer-side rescoring with HF transformers (the usual policy forward in an RL trainer).

  python score_hf.py --rollout R.json --model BF16_DIR --data GSM_DIR --dtype bf16 --bs 8 --order group --out S.json

Teacher forcing on prompt + the exact sampled tokens, right-padded batches. Batch order: group (the G = 8
samples of one prompt together, GRPO-style), sorted (by total length), shuffle (fixed RNG). Per token:
log p (log-softmax in float32 of the model's logits) and argmax. fp32 runs with TF32 off.
"""

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def batches(rs, prompts, bs, order):
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
    if order == "sorted":
        items.sort(key=lambda x: len(x[1]) + len(x[2]))
    elif order == "shuffle":
        random.Random(11).shuffle(items)
    return [items[i:i + bs] for i in range(0, len(items), bs)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--order", choices=["group", "sorted", "shuffle"], default="group")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation=a.attn).cuda().eval()
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    out, t0 = {}, time.time()
    for b in batches(rs, prompts, a.bs, a.order):
        L = max(len(p) + len(r) for _, p, r in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, (_, p, r) in enumerate(b):
            ids[i, :len(p) + len(r)] = torch.tensor(p + r)
            mask[i, :len(p) + len(r)] = 1
        h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
        for i, (s, p, r) in enumerate(b):
            P, n = len(p), len(r)
            lg = model.lm_head(h[i, P - 1:P - 1 + n]).float().log_softmax(-1)
            y = torch.tensor(r, device=lg.device)
            out[s] = {"lp": lg[torch.arange(n), y].tolist(), "am": lg.argmax(-1).tolist()}
    meta = dict(vars(a), sec=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__)
    common.save(a.out, f"hf_{a.dtype}", meta, out)
    print(f"RESULT {a.out} seqs {len(out)} sec {meta['sec']}")


if __name__ == "__main__":
    main()
