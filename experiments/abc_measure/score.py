"""Trainer side of the A/B/C measurement: teacher-forced log p of a rollout's tokens, any length.

  python score.py --mode hf_bf16|hf_fp32|engine --rollout R.json --model HF_DIR --data GSM_DIR --out S.json
                  [--const C.pt] [--bs 8] [--chunk 256]

hf_*    : HF transformers (sdpa), model dtype bf16 / fp32 (TF32 off), model.model() then lm_head in chunks
          of --chunk positions, float32 log-softmax (as rl_mismatch/score_hf.py).
engine  : BitExact standalone int8 engine. Sequences are prefilled in chunks of --chunk positions through
          the integer KV cache (same integers as one full forward; checked against the plugin's logits SHA);
          records sha16 of every int64 logits row and the canonical float64 log p (common.int_logp).
Batches are right-padded groups of --bs sequences in seed order (bs 1 for the long runs).
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from bitexact_engine import _pad, _rows  # noqa: E402


def load_engine(a, items):
    """IntLlama from the constants, RoPE rows extended (same rule, bit-identical first 2048 rows) to the longest sequence."""
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin.model import _extend_rope
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model, max(len(p) + len(r) for _, p, r in items) + 8)
    return IntLlama(c, "cuda")


@torch.no_grad()
def hf_score(a, items):
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dt = torch.bfloat16 if a.mode == "hf_bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").cuda().eval()
    out = {}
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]
        L = max(len(p) + len(r) for _, p, r in b)
        ids = torch.full((len(b), L), 0, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for j, (_, p, r) in enumerate(b):
            ids[j, :len(p) + len(r)] = torch.tensor(p + r)
            mask[j, :len(p) + len(r)] = 1
        h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda() if len(b) > 1 else None).last_hidden_state
        for j, (s, p, r) in enumerate(b):
            P, n = len(p), len(r)
            lps, ams = [], []
            for c0 in range(0, n, a.chunk):
                c1 = min(n, c0 + a.chunk)
                lg = model.lm_head(h[j, P - 1 + c0:P - 1 + c1]).float().log_softmax(-1)
                y = torch.tensor(r[c0:c1], device=lg.device)
                lps += lg[torch.arange(c1 - c0), y].tolist()
                ams += lg.argmax(-1).tolist()
            out[s] = {"lp": lps, "am": ams}
        del h
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def engine_score(a, items):
    m = load_engine(a, items)
    out = {}
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]
        ids, pos, L = _pad(m, [p + r for _, p, r in b])
        cache = m.new_cache(len(b), L)
        hs = []
        for c0 in range(0, L, a.chunk):
            c1 = min(L, c0 + a.chunk)
            hs.append(m.forward(ids[:, c0:c1], pos[:, c0:c1], cache, S=c1))
        h = torch.cat(hs, 1)
        del hs, cache
        for j, (s, p, r) in enumerate(b):
            P, n = len(p), len(r)
            sha, am, lp = [], [], []
            for c0 in range(0, n, a.chunk):
                c1 = min(n, c0 + a.chunk)
                x, y, z = _rows(m.logits(h[j, P - 1 + c0:P - 1 + c1]), r[c0:c1])
                sha += x
                am += y
                lp += z
            out[s] = {"sha": sha, "am": am, "lp": lp}
        m.check()
        del h
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf_bf16", "hf_fp32", "engine"], required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--model", required=True, help="HF dir (hf modes; tokenizer for both)")
    ap.add_argument("--const")
    ap.add_argument("--data", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
    t0 = time.time()
    out = engine_score(a, items) if a.mode == "engine" else hf_score(a, items)
    meta = dict(vars(a), sec=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    common.save(a.out, a.mode, meta, out)
    print(f"RESULT {a.out} seqs {len(out)} sec {meta['sec']}")


if __name__ == "__main__":
    main()
