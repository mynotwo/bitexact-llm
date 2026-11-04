"""cuBLASLt int8 via ctypes: time every heuristic algorithm for C[M,N] = a[M,K] @ w[N,K]^T (torch._int_mm layout)."""

import ctypes as C
import glob
import json
import sys

import torch
import triton

lib = C.CDLL(
    glob.glob("/home/audreymao/abc_measure/venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13")[0]
)


def ck(s, what):
    if s != 0:
        raise RuntimeError(f"{what} status {s}")


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heur(C.Structure):
    _fields_ = [("algo", Algo), ("ws", C.c_size_t), ("state", C.c_int), ("waves", C.c_float), ("res", C.c_int * 4)]


R_8I, R_32I, COMPUTE_32I = 3, 10, 72
h = C.c_void_p()
ck(lib.cublasLtCreate(C.byref(h)), "create")
WS = 32 << 20
ws = torch.empty(WS, dtype=torch.uint8, device="cuda")


def setup(M, N, K):
    d = C.c_void_p()
    ck(lib.cublasLtMatmulDescCreate(C.byref(d), COMPUTE_32I, R_32I), "desc")
    t = C.c_int(1)
    ck(lib.cublasLtMatmulDescSetAttribute(d, 3, C.byref(t), 4), "transa")
    lay = []
    for typ, r, c, ld in ((R_8I, K, N, K), (R_8I, K, M, K), (R_32I, N, M, N)):
        lo = C.c_void_p()
        ck(lib.cublasLtMatrixLayoutCreate(C.byref(lo), typ, C.c_uint64(r), C.c_uint64(c), C.c_int64(ld)), "lay")
        lay.append(lo)
    p = C.c_void_p()
    ck(lib.cublasLtMatmulPreferenceCreate(C.byref(p)), "pref")
    w = C.c_uint64(WS)
    ck(lib.cublasLtMatmulPreferenceSetAttribute(p, 1, C.byref(w), 8), "prefws")
    hs = (Heur * 32)()
    n = C.c_int()
    ck(lib.cublasLtMatmulAlgoGetHeuristic(h, d, lay[0], lay[1], lay[2], lay[2], p, 32, hs, C.byref(n)), "heur")
    return d, lay, [hs[i] for i in range(n.value) if hs[i].state == 0]


def run(d, lay, hr, a, w, c):
    al, be = C.c_int(1), C.c_int(0)
    ck(
        lib.cublasLtMatmul(
            h,
            d,
            C.byref(al),
            C.c_void_p(w.data_ptr()),
            lay[0],
            C.c_void_p(a.data_ptr()),
            lay[1],
            C.byref(be),
            C.c_void_p(c.data_ptr()),
            lay[2],
            C.c_void_p(c.data_ptr()),
            lay[2],
            C.byref(hr.algo),
            C.c_void_p(ws.data_ptr()),
            C.c_size_t(WS),
            C.c_void_p(torch.cuda.current_stream().cuda_stream),
        ),
        "mm",
    )


def kname(f):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f()
        torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA][:1]


SHAPES = {
    "qkv": (3072, 2048),
    "o": (2048, 2048),
    "gate_up": (16384, 2048),
    "down": (2048, 8192),
    "lm_head": (128256, 2048),
}
rows = []
for name, (N, K) in SHAPES.items():
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    for M in (24, 128, 1024, 4096, 16384):
        if name == "lm_head" and M > 4096:
            continue
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        ref = torch._int_mm(a, w.t())
        tr = triton.testing.do_bench(lambda: torch._int_mm(a, w.t()), warmup=10, rep=50)
        d, lay, hs = setup(M, N, K)
        best = None
        for i, hr in enumerate(hs):
            c = torch.empty(M, N, dtype=torch.int32, device="cuda")
            try:
                run(d, lay, hr, a, w, c)
            except RuntimeError:
                continue
            torch.cuda.synchronize()
            eq = bool(torch.equal(c, ref))
            t = triton.testing.do_bench(lambda: run(d, lay, hr, a, w, c), warmup=10, rep=50)
            if best is None or t < best[0]:
                best = (t, i, eq, kname(lambda: run(d, lay, hr, a, w, c)))
        ops = 2 * M * N * K
        row = dict(
            name=name,
            M=M,
            N=N,
            K=K,
            intmm_us=tr * 1e3,
            intmm_tops=ops / tr / 1e9,
            n_algos=len(hs),
            lt_best_us=best[0] * 1e3,
            lt_best_tops=ops / best[0] / 1e9,
            lt_best_idx=best[1],
            lt_equal=best[2],
            lt_kernel=best[3],
            intmm_kernel=kname(lambda: torch._int_mm(a, w.t())),
        )
        rows.append(row)
        print(row, flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
