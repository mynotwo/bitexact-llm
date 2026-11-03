"""Bound stress for the limb-stacked decode kernels: ~43.7k keys, |q|,|K|,|V| at LIM[3] extremes, equal scores
(every key gets the max weight), compared with the CPU int64 reference and the general kernels."""
import torch
from bitexact.int8.consts import exp_table
from bitexact.int8.segs import softmax
from bitexact.intops import LIM
from bitexact.vllm_plugin import tri_paged
L = LIM[3]; print("LIM3", L, "neg all-limb -128:", -128 * (1 + 256 + 65536))
g = torch.Generator(device="cuda").manual_seed(1)
lut = exp_table().cuda(); tab = tri_paged.weight_table(lut); print("ptab max", int(tab.max()))
bs, HKV, HD, W = 16, 8, 64, 2730          # W * bs = 43680 < 43690
nb = W + 1
for case in ("neg", "pos", "pm", "kneg_vpm"):
    for R, M in [(1, 4), (2, 1), (1, 3)]:
        P = W * bs
        if case == "neg":   c = torch.full((nb, HKV, bs, 2 * HD), -L, dtype=torch.int32, device="cuda"); q = torch.full((R, HKV, M, HD), -L, device="cuda")
        elif case == "pos": c = torch.full((nb, HKV, bs, 2 * HD), L, dtype=torch.int32, device="cuda"); q = torch.full((R, HKV, M, HD), L, device="cuda")
        else:
            c = (torch.randint(0, 2, (nb, HKV, bs, 2 * HD), device="cuda", generator=g) * 2 - 1).to(torch.int32) * L
            q = (torch.randint(0, 2, (R, HKV, M, HD), device="cuda", generator=g) * 2 - 1) * L
            if case == "kneg_vpm": c[..., :HD] = -L; q[:] = -L
        bt = torch.arange(1, nb, device="cuda", dtype=torch.int32).repeat(R, 1)
        qpos = torch.full((R, M), P - 1, device="cuda"); qpos[:, 0] -= 7
        S = P; s = torch.arange(S, device="cuda")
        kv = c[bt.long()[:, s // bs], :, (s % bs)[None].expand(R, -1)].long()
        K, V = kv[..., :HD].permute(0, 2, 1, 3).cpu(), kv[..., HD:].permute(0, 2, 1, 3).cpu()
        raw = torch.einsum("rhmd,rhsd->rhms", q.cpu(), K)
        p, den = softmax(raw, qpos.cpu()[:, None, :, None], lut.cpu())
        ref = torch.einsum("rhms,rhsd->rhmd", p, V)
        for ns in (None, 1, 3, 64):
            for bm in (None, 16):
                num, den2 = tri_paged.attend(q, qpos, tri_paged.to_limbs(c), bt, tab, nsplit=ns, bm=bm)
                ok = torch.equal(num.cpu(), ref) and torch.equal(den2.cpu(), den)
                assert ok, (case, R, M, ns, bm)
        print(case, R, M, "ok; max|ref num|", int(ref.abs().max()), "den", int(den.max()))
print("EXTREME_OK")
