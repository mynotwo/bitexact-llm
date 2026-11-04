"""int8 GEMM microbench: torch._int_mm vs _scaled_mm vs Triton tl.dot (sm90) on Llama-3.2-1B linear shapes."""

import json
import sys

import torch
import triton
import triton.language as tl

SHAPES = {
    "qkv": (3072, 2048),
    "o": (2048, 2048),
    "gate_up": (16384, 2048),
    "down": (2048, 8192),
    "lm_head": (128256, 2048),
}
MS = [24, 64, 128, 512, 1024, 2048, 4096, 8192, 16384]


def cfgs():
    out = []
    for bm in (64, 128):
        for bn in (64, 128, 256):
            for bk in (64, 128, 256):
                for st in (3, 4, 5):
                    for w in (4, 8):
                        if bm * bk + bn * bk > 160 * 1024 // st:
                            continue
                        out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "G": 8}, num_stages=st, num_warps=w))
    return out


@triton.autotune(configs=cfgs(), key=["M", "N", "K"])
@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (G * npn)
    fm = gid * G
    gs = min(npm - fm, G)
    pm = fm + (pid % (G * npn)) % gs
    pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0)
        y = tl.load(b, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0)
        acc = tl.dot(x, y, acc, out_dtype=tl.int32)
        a += BK * sak
        b += BK * sbk
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def tri_mm(a, wt):  # a [M,K] int8, wt = w.t() [K,N] (w [N,K] row-major -> column-major view)
    M, K = a.shape
    N = wt.shape[1]
    c = torch.empty(M, N, dtype=torch.int32, device=a.device)
    grid = lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)
    mm_kernel[grid](a, wt, c, M, N, K, a.stride(0), a.stride(1), wt.stride(0), wt.stride(1), c.stride(0), c.stride(1))
    return c


def tbench(f, n=50):
    f()
    torch.cuda.synchronize()
    return triton.testing.do_bench(f, warmup=10, rep=n)  # ms, median


def main():
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    W = {k: torch.randint(-127, 128, (n, kk), dtype=torch.int8, device=dev) for k, (n, kk) in SHAPES.items()}
    # scaled_mm int8 support probe
    sm_ok = None
    try:
        a = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device=dev)
        b = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device=dev).t().contiguous().t()
        r = torch._scaled_mm(a, b, torch.tensor(1.0, device=dev), torch.tensor(1.0, device=dev), out_dtype=torch.int32)
        sm_ok = bool(torch.equal(r, torch._int_mm(a, b)))
    except Exception as e:
        sm_ok = f"unsupported: {type(e).__name__}: {str(e)[:200]}"
    print("scaled_mm int8:", sm_ok, flush=True)
    wg = None
    for name, w in W.items():
        N, K = w.shape
        for M in MS:
            if name == "lm_head" and M > 4096:
                continue
            a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=dev)
            a[0, :] = -128
            w[0, :] = -128 if name != "x" else w[0, :]  # extreme-value row/col for the bound
            wt = w.t()
            ref = torch._int_mm(a, wt)
            t_ref = tbench(lambda: torch._int_mm(a, wt))
            c = tri_mm(a, wt)
            eq = bool(torch.equal(c, ref))
            t_tri = tbench(lambda: tri_mm(a, wt))
            best = mm_kernel.best_config
            if wg is None:
                k = list(mm_kernel.cache.values())[0] if hasattr(mm_kernel, "cache") else None
            ops = 2 * M * N * K
            row = dict(
                name=name,
                M=M,
                N=N,
                K=K,
                intmm_us=t_ref * 1e3,
                intmm_tops=ops / t_ref / 1e9,
                tri_us=t_tri * 1e3,
                tri_tops=ops / t_tri / 1e9,
                speedup=t_ref / t_tri,
                tri_equal=eq,
                cfg=str(best),
            )
            rows.append(row)
            print(
                f"{name:8s} M={M:6d} N={N:6d} K={K:5d} int_mm {row['intmm_us']:9.1f}us {row['intmm_tops']:6.0f}T | "
                f"triton {row['tri_us']:9.1f}us {row['tri_tops']:6.0f}T x{row['speedup']:.2f} eq={eq} {best}",
                flush=True,
            )
    # wgmma check on compiled kernels
    asm = []
    for k in mm_kernel.fn.device_caches[0][0].values() if hasattr(mm_kernel.fn, "device_caches") else []:
        try:
            asm.append("wgmma" in k.asm["ptx"])
        except Exception:
            pass
    print("ptx wgmma in compiled kernels:", sum(asm), "/", len(asm))
    json.dump(dict(scaled_mm=sm_ok, wgmma=[sum(asm), len(asm)], rows=rows), open(sys.argv[1], "w"), indent=1)


main()
