"""Score the real workload at TP 1 / 2 / 4: 32 GSM8K prompts x G 8 seeded requests, teacher forcing.

  torchrun --nproc_per_node=N tp_gsm.py --const C --tokenizer DIR --data GSM_DIR --out J [--max-prompts 32]

Unlike tp_run.py (greedy replay on 4 fixed prompts), this scores rollout-fixed token histories in
right-padded batches, which is what the trainer does. Every rank computes locally; rank 0 gathers the
vocabulary shard and records sha16 / argmax / int_logp per position, through the same code path as
`experiments/rl_mismatch/bitexact_engine.py score`.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402
from tpmodel import TPIntLlama, gather_logits, shard  # noqa: E402


def rows_of(lg, toks):
    am = torch.argmax(lg, -1).tolist()
    arr = lg.cpu().numpy()
    sha = [hashlib.sha256(arr[i].tobytes()).hexdigest()[:16] for i in range(len(toks))]
    return sha, am, [common.int_logp(arr[i], toks[i]) for i in range(len(toks))]


@torch.no_grad()
def score(m, items, bs, world, dev, rank):
    from bitexact.llama import PAD
    out = {}
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        seqs = [p + r for _, p, r in b]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        ids = ids.to(dev)
        pos = torch.arange(L, device=dev).expand(len(seqs), L).contiguous()
        h = m.forward(ids, pos, m.new_cache(len(seqs), L))
        for j, (s, p, r) in enumerate(b):
            P, n = len(p), len(r)
            lg = m.logits(h[j, P - 1:P - 1 + n])
            full = gather_logits(lg, world, dev)
            if rank == 0:
                sha, am, lp = rows_of(full, r)
                out[s] = {"sha": sha, "am": am, "lp": lp}
        if rank == 0:
            print(f"  batch {i // bs + 1}/{(len(items) + bs - 1) // bs}", flush=True)
    m.check()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rollout", required=True, help="rollout JSON with the token histories to score")
    ap.add_argument("--max-prompts", type=int, default=32)
    ap.add_argument("--tp", type=int, default=0, help="only used in the output name when not launched by torchrun")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    n = torch.cuda.device_count()
    torch.cuda.set_device(rank % n)
    dev = f"cuda:{rank % n}"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, n=a.max_prompts)
    reqs = common.requests(prompts)
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    m = TPIntLlama(shard(c, rank, world), dev, rank, world)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
    t0 = __import__("time").time()
    out = score(m, items, a.bs, world, dev, rank)
    sec = __import__("time").time() - t0
    if rank == 0:
        meta = {"engine": f"bitexact_tp{world}", "tp": world, "bs": a.bs, "sec": round(sec, 1),
                "const_sha": m.sha, "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
                "rollout": a.rollout}
        common.save(a.out, f"bitexact_tp{world}_score", meta, out)
        print(f"RESULT {a.out} seqs {len(out)} sec {meta['sec']}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
