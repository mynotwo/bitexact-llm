import itertools, sys, json, torch, triton
from bitexact.int8.gemm_skinny import gemm
SH = {"qkv": (3072, 2048), "o": (2048, 2048), "gu": (16384, 2048), "down": (2048, 8192), "lm": (128256, 2048)}
ROWS = [24, 64, 256, 384]
def t(f, n=50):
    f(); torch.cuda.synchronize()
    return triton.testing.do_bench(f, warmup=10, rep=60) * 1e3  # us
res = {}
for name, (N, K) in SH.items():
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    for R in ROWS:
        a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda")
        ref = torch._int_mm(a, w.t())
        tb = t(lambda: torch._int_mm(a, w.t()))
        roof = N * K / 3.35e12 * 1e6
        best = (1e9, None)
        BMs = [16, 32] if R <= 64 else [64, 128]
        for BM, BN, BK, S, nw, ns in itertools.product(BMs, [64, 128, 256], [128, 256], [1, 2, 4], [4, 8], [3, 4]):
            if BM > max(16, R) * 2 or K % (BK * S): continue
            if R > 64 and BM * (R // BM) < R and False: continue
            if (BM + BN) * BK * ns > 200000: continue
            try:
                cfg = (BM, BN, BK, S, nw, ns)
                if not torch.equal(gemm(a, w, cfg), ref): print("MISMATCH", cfg); continue
                x = t(lambda: gemm(a, w, cfg))
            except Exception as e:
                if 'ERR' not in res: res['ERR']=repr(e)[:200]; print('ERR', repr(e)[:300])
                continue
            if x < best[0]: best = (x, cfg)
        print(f"{name} R={R} int_mm={tb:.1f}us triton={best[0]:.1f}us {best[1]} roof={roof:.1f}us", flush=True)
        res[f"{name},{R}"] = dict(int_mm=tb, tri=best[0], cfg=best[1], roof=roof)
json.dump(res, open(sys.argv[1], "w"), indent=1)
