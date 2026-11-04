"""cuBLASLt int8 x int8 -> int32 GEMM with a workspace (sm90 wgmma kernels), bit-identical to torch._int_mm.

Used by gemm_skinny.int_mm for every shape its skinny Triton table does not cover (prefill / scorer rows > 256,
the LM head). From exp/speed-gemm (experiments/speed_gemm/RESULTS.md there).
"""

import ctypes
import glob
import os
import sys

import torch

# ---- int8 x int8 -> int32 GEMM for the linear layers ----
# On H100 torch._int_mm's cuBLASLt call runs the sm80 mma.sync kernel (cutlass_80_tensorop_i16832gemm_s8).
# Called here with a 32 MiB workspace, the cuBLASLt heuristic offers sm90 wgmma kernels (sm90_xmma_gemm_i8i32_*,
# some split-K / stream-K) that are up to 2x faster on prefill shapes. Every algorithm sums int8 products in int32
# with an integer epilogue (alpha = 1, beta = 0): a partial sum over any subset of k is bounded by
# K * 128 * 128 < 2^31 for K < 2^17, and int32 addition is exact, so all algorithms return the same integers.
# Default on CUDA (BITEXACT_GEMM=intmm selects torch._int_mm). On the first call of each shape (outside CUDA graph
# capture) the heuristic's candidates and torch._int_mm are timed and the fastest is kept; the choice only
# affects speed. Falls back to torch._int_mm for operands it does not cover.

GEMM = os.environ.get("BITEXACT_GEMM", "lt" if torch.cuda.is_available() else "intmm")
_LT = {}                                                    # lib, handle, per-device workspaces, plans
_WS = 32 << 20
_NALGO = 8
_R8I, _R32I, _C32I = 3, 10, 72                              # cudaDataType / cublasComputeType values


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("ws", ctypes.c_size_t), ("state", ctypes.c_int), ("waves", ctypes.c_float),
                ("res", ctypes.c_int * 4)]


def _ck(s, what):
    if s != 0:
        raise RuntimeError(f"cuBLASLt {what}: status {s}")


def _lib():
    if "lib" not in _LT:
        name = f"libcublasLt.so.{torch.version.cuda.split('.')[0]}"
        try:
            lib = ctypes.CDLL(name)                         # the copy torch already loaded
        except OSError:
            hits = [p for d in sys.path for p in glob.glob(os.path.join(d, "nvidia", "*", "lib", name))]
            lib = ctypes.CDLL(hits[0])
        h = ctypes.c_void_p()
        _ck(lib.cublasLtCreate(ctypes.byref(h)), "create")
        _LT["lib"], _LT["h"] = lib, h
    return _LT["lib"]


def _align(p):
    return min(256, p & -p) if p else 256


def _plan(M, N, K, lda, aligns):
    key = (M, N, K, lda, aligns)
    if key in _LT:
        return _LT[key]
    lib, c = _lib(), ctypes
    d = c.c_void_p()
    _ck(lib.cublasLtMatmulDescCreate(c.byref(d), _C32I, _R32I), "desc")
    t = c.c_int(1)                                          # CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_T
    _ck(lib.cublasLtMatmulDescSetAttribute(d, 3, c.byref(t), 4), "transa")
    lay = []                                                # column-major view: C^T [N, M] = W [K, N]^T a^T [K, M]
    for typ, r, cols, ld in ((_R8I, K, N, K), (_R8I, K, M, lda), (_R32I, N, M, N)):
        lo = c.c_void_p()
        _ck(lib.cublasLtMatrixLayoutCreate(c.byref(lo), typ, c.c_uint64(r), c.c_uint64(cols), c.c_int64(ld)), "layout")
        lay.append(lo)
    p = c.c_void_p()
    _ck(lib.cublasLtMatmulPreferenceCreate(c.byref(p)), "pref")
    ws = c.c_uint64(_WS)
    _ck(lib.cublasLtMatmulPreferenceSetAttribute(p, 1, c.byref(ws), 8), "pref ws")
    for attr, al in zip((5, 6, 7, 8), aligns):              # MIN_ALIGNMENT_{A,B,C,D}_BYTES
        v = c.c_uint32(al)
        _ck(lib.cublasLtMatmulPreferenceSetAttribute(p, attr, c.byref(v), 4), "pref align")
    hs, n = (_Heur * _NALGO)(), c.c_int()
    _ck(lib.cublasLtMatmulAlgoGetHeuristic(_LT["h"], d, lay[0], lay[1], lay[2], lay[2], p, _NALGO, hs,
                                           c.byref(n)), "heuristic")
    algos = [hs[i] for i in range(n.value) if hs[i].state == 0 and hs[i].ws <= _WS]
    _LT[key] = [d, lay, algos, None]                        # choice None: not timed yet (heuristic's first)
    return _LT[key]


def _time(f, n=5):
    f()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)


def _choose(plan, a, w, out):
    """Index of the fastest candidate on these operands; -1 = torch._int_mm. Every candidate writes into the caller's
    out (no extra [M, N] buffer: vLLM's memory profile would count it and shrink the KV cache)."""
    best = (_time(lambda: torch._int_mm(a, w.t(), out=out)), -1)
    for i, hr in enumerate(plan[2]):
        try:
            best = min(best, (_time(lambda: _run(plan, hr, a, w, out)), i))
        except RuntimeError:
            continue
    return best[1]


def _run(plan, hr, a, w, out):
    d, lay = plan[0], plan[1]
    dev = a.device.index if a.device.index is not None else torch.cuda.current_device()
    wsk = ("ws", dev)
    if wsk not in _LT:
        _LT[wsk] = torch.empty(_WS, dtype=torch.uint8, device=a.device)
    c = ctypes
    al, be = c.c_int32(1), c.c_int32(0)
    stream = c.c_void_p(torch.cuda.current_stream(a.device).cuda_stream)
    _ck(_lib().cublasLtMatmul(_LT["h"], d, c.byref(al), c.c_void_p(w.data_ptr()), lay[0], c.c_void_p(a.data_ptr()),
                              lay[1], c.byref(be), c.c_void_p(out.data_ptr()), lay[2], c.c_void_p(out.data_ptr()),
                              lay[2], c.byref(hr.algo), c.c_void_p(_LT[wsk].data_ptr()), c.c_size_t(_WS), stream),
        "matmul")


def _lt_mm_impl(a, w):
    M, K = a.shape
    N = w.shape[0]
    assert K < 1 << 17                                      # |partial sum| <= K * 2^14 < 2^31
    out = torch.empty((M, N), dtype=torch.int32, device=a.device)
    aligns = (_align(w.data_ptr()), _align(a.data_ptr()), _align(out.data_ptr()), _align(out.data_ptr()))
    plan = _plan(M, N, K, a.stride(0), aligns)
    if plan[3] is None and not torch.cuda.is_current_stream_capturing():
        plan[3] = _choose(plan, a, w, out) if plan[2] else -1
    i = 0 if plan[3] is None else plan[3]
    if i < 0 or not plan[2]:
        return torch._int_mm(a, w.t(), out=out)
    _run(plan, plan[2][i], a, w, out)
    return out


@torch.library.custom_op("bitexact::lt_int_mm", mutates_args=())
def lt_int_mm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a int8 [M, K] (row stride >= K), w int8 [N, K] contiguous -> exact int32 a @ w^T via cuBLASLt."""
    return _lt_mm_impl(a, w)


@lt_int_mm.register_fake
def _(a, w):
    return a.new_empty((a.shape[0], w.shape[0]), dtype=torch.int32)


def int_mm(a, w, kind=None):
    """Exact int32 a @ w^T for int8 a [M, K], w [N, K] on CUDA (torch._int_mm shape rules)."""
    if (kind or GEMM) == "lt" and a.stride(1) == 1 and w.is_contiguous():
        return lt_int_mm(a, w)
    return torch._int_mm(a, w.t())
