"""Bisect a TP mismatch: compare the hidden state and logits after each stage of one forward."""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitexact.int8.model import IntLlama
import tpmodel

@torch.no_grad()
def record(c, dev, rank, world, ids):
    m = tpmodel.TPIntLlama(tpmodel.shard(c, rank, world), dev, rank, world)
    B, T = ids.shape
    pos = torch.arange(T, device=dev)[None].expand(B, T).contiguous()
    cache = m.new_cache(B, T)
    x = m.emb[ids].long()
    import hashlib
    H = lambda t: hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:12]
    rec = {"emb": H(x)}
    L, Dd = m.L, 2048
    for i in range(L):
        xa = m.attention(i, x, pos, cache, T)
        if rank == 0:
            rec[f"att{i}"] = H(xa)
        x = m.mlp(i, xa)
        if rank == 0:
            rec[f"mlp{i}"] = H(x)
    lg = m.logits(x[:, -1:].reshape(-1, Dd))
    full = tpmodel.gather_logits(lg, world, dev)
    if rank == 0:
        rec["logits"] = H(full)
    return rec

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
dev = f"cuda:{rank % torch.cuda.device_count()}"
c = torch.load("/data41/audreymao/bitexact_port/data/intllama_const.pt", map_location="cpu", weights_only=False)
ids = torch.tensor([[128000, 791, 23070, 19044]], device=dev)
r = record(c, dev, rank, world, ids)
if rank == 0:
    print("WORLD", world)
    for k, v in r.items():
        print(f"  {k:8} {v}")
dist.barrier(); dist.destroy_process_group()
