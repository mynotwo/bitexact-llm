"""Profiling helpers for the int8 engine: category annotations, chrome-trace attribution, logical bytes.

annotate(m): wraps the engine's segments, GEMMs and attention products in torch.profiler record_function
ranges named "cat:<name>". Only meaningful in eager mode (compiled=False): inside a torch.compile region the
inner ranges are dropped, so compiled runs are attributed by kernel name instead (kernel_class).

attribute(trace.json): every GPU kernel is attributed to the innermost "cat:" range that contains its launch
(cudaLaunchKernel & co, matched by correlation id) and to the innermost aten op (for GEMM shapes). Returns
per-category kernel time, kernel counts, GPU busy time (union of kernel intervals) and the wall span.

ByteCounter: a TorchDispatchMode that adds input + output bytes (unique storages) of every aten op to the
current category: the traffic of eager ops if each reads its inputs and writes its output once from DRAM.
Triton kernels launched directly (bitexact.bmm, tri_paged) are invisible to it; their bytes are added by the
callers' wrappers (inputs + output once).
"""

import collections
import contextlib
import json

import torch
from torch.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode

STACK = []          # category stack shared by the profiler ranges and the byte counter


@contextlib.contextmanager
def cat(name):
    STACK.append(name)
    try:
        with record_function("cat:" + name):
            yield
    finally:
        STACK.pop()


def _wrap(fn, name):
    def f(*a, **k):
        with cat(name):
            return fn(*a, **k)
    f.__wrapped__ = fn
    return f


def _nbytes(t):
    return t.numel() * t.element_size()


def annotate(m, fine=True):
    """Wrap the int8 engine m (IntLlama) in place. fine=False (compiled segments): only the segment-level ranges,
    which sit outside the compiled functions; the fine ranges would be inside them."""
    import bitexact.int8.model as M
    import bitexact.int8.segs as S

    # fine ranges inside the segments (module globals looked up at call time)
    for fname, cname in [("rms", "rmsnorm"), ("quant_rows", "quant"), ("norm24", "quant"), ("stack_limbs", "limb_split"),
                         ("join_limbs", "limb_join"), ("silu", "silu_lut"), ("exp_of", "exp_lut"), ("_rope", "rope")]:
        if not fine:
            break
        g = getattr(S, fname)
        g = getattr(g, "__wrapped__", g)
        setattr(S, fname, _wrap(g, cname))
    for aname, cname in [("pre_norm", "pre_norm"), ("pre_plain", "pre_plain"), ("post", "post"), ("post_add", "post"),
                         ("qkv", "qkv_rope"), ("softmax", "softmax"), ("attn_out", "attn_out"), ("silu_mul", "silu_mul")]:
        setattr(m.sg, aname, _wrap(getattr(m.sg, aname), cname))

    gemm = m._gemm
    m._gemm = _wrap(gemm, "gemm")
    logits = m.logits
    m.logits = _wrap(logits, "lm_head")
    cw = M.Cache.write
    M.Cache.write = _wrap(getattr(cw, "__wrapped__", cw), "kv_write")

    bmm0 = getattr(M.bmm, "__wrapped__", M.bmm)
    state = {"n": 0}

    def bmm(a, b, *x, **k):
        name = "attn_qk" if state["n"] % 2 == 0 else "attn_pv"
        state["n"] += 1
        with cat(name):
            c = bmm0(a, b, *x, **k)
        return c
    bmm.__wrapped__ = bmm0
    M.bmm = bmm


class ByteCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.bytes = collections.Counter()
        self.ops = collections.Counter()

    def add(self, name, n):
        self.bytes[name] += n

    SKIP = ("empty", "empty_like", "empty_strided", "_local_scalar_dense", "lift_fresh")
    GATHER = ("index", "embedding", "index_select", "gather", "take")
    SCATTER = ("index_put_", "index_put", "_index_put_impl_", "scatter_", "scatter", "index_copy_")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func.__name__.split(".")[0] in self.SKIP:
            return out
        leaves = lambda x: [t for t in torch.utils._pytree.tree_leaves(x) if torch.is_tensor(t) and t.is_cuda]
        ins, outs = leaves((args, kwargs)), leaves(out)
        in_ptrs = {t.untyped_storage().data_ptr() for t in ins}
        if outs and all(t.untyped_storage().data_ptr() in in_ptrs for t in outs) and not func.__name__.split(".")[0].endswith("_"):
            return out                                          # a view (or a no-op): no traffic
        base = func.__name__.split(".")[0]
        if base in self.GATHER:                                 # reads only the gathered rows, not the whole table
            idx = [t for t in ins[1:] if not t.is_floating_point() and t.dtype in (torch.long, torch.int32, torch.bool)]
            n = 2 * sum(_nbytes(t) for t in outs) + sum(_nbytes(t) for t in idx)
            self.bytes["/".join(STACK) or "(none)"] += n
            self.ops["/".join(STACK) or "(none)"] += 1
            return out
        if base in self.SCATTER:                                # writes only the indexed rows of the destination
            n = 2 * sum(_nbytes(t) for t in ins[1:])
            self.bytes["/".join(STACK) or "(none)"] += n
            self.ops["/".join(STACK) or "(none)"] += 1
            return out
        seen, n = set(), 0
        for t in ins + outs:
            key = (t.untyped_storage().data_ptr(), t.storage_offset(), t.numel())
            if key not in seen and t.numel() > 1:
                seen.add(key)
                n += min(_nbytes(t), t.untyped_storage().nbytes())      # expanded (stride-0) inputs read once
        path = "/".join(STACK) or "(none)"
        self.bytes[path] += n
        self.ops[path] += 1
        return out


LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaGraphLaunch")


def attribute(path):
    """Chrome trace -> dict(by_path {annotation path: [kernel us, kernel count]}, kernels [(name, dur, path, op,
    dims)], busy_us, span_us, n_kernels)."""
    ev = json.load(open(path))["traceEvents"]
    ranges = collections.defaultdict(list)          # tid -> [(ts, end, name, kind, dims)]
    launches = {}
    kernels = []
    ext_op = {}                                     # External id -> its cpu_op (kernels without a traced launch)
    for e in ev:
        c, ph = e.get("cat", ""), e.get("ph")
        if ph != "X":
            continue
        if c in ("user_annotation", "cpu_op"):
            ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"], c, e.get("args", {}).get("Input Dims")))
            if c == "cpu_op" and "External id" in e.get("args", {}):
                ext_op[e["args"]["External id"]] = (e["tid"], e["ts"] + 1e-3)
        elif c in ("cuda_runtime", "cuda_driver") and e["name"] in LAUNCH:
            launches[e["args"]["correlation"]] = (e["tid"], e["ts"])
        elif c == "kernel":
            kernels.append(e)
    for k in kernels:                               # e.g. cuBLASLt/CUTLASS launches CUPTI does not report
        corr, ext = k["args"].get("correlation"), k["args"].get("External id")
        if corr not in launches and ext in ext_op:
            launches[("ext", ext)] = ext_op[ext]
            k["args"]["correlation"] = ("ext", ext)
    # per thread: sweep launches in time order with a stack of open ranges (ranges nest properly per thread)
    by_tid = collections.defaultdict(list)
    for corr, (tid, ts) in launches.items():
        by_tid[tid].append((ts, corr))
    ctx = {}
    for tid, ls in by_tid.items():
        ls.sort()
        rs = sorted(ranges.get(tid, []), key=lambda r: (r[0], -r[1]))
        stack, j = [], 0
        for ts, corr in ls:
            while j < len(rs) and rs[j][0] <= ts:
                while stack and stack[-1][1] < rs[j][0]:
                    stack.pop()
                stack.append(rs[j])
                j += 1
            while stack and stack[-1][1] < ts:
                stack.pop()
            enc = [r for r in stack if r[1] >= ts]
            cats = [r[2][4:] for r in enc if r[3] == "user_annotation" and r[2].startswith("cat:")]
            ops = [r for r in enc if r[3] == "cpu_op"]
            anns = [r[2] for r in enc if r[3] == "user_annotation" and not r[2].startswith("cat:")]
            ctx[corr] = ("/".join(cats) or "(none)", ops[-1][2] if ops else None, ops[-1][4] if ops else None,
                         anns[-1] if anns else None)
    by_path = collections.defaultdict(lambda: [0.0, 0])
    rows = []
    iv = []
    for k in kernels:
        corr = k.get("args", {}).get("correlation")
        path, op, dims, ann = ctx.get(corr, ("(unattributed)", None, None, None))
        by_path[path][0] += k["dur"]
        by_path[path][1] += 1
        rows.append((k["name"], k["dur"], path, op, dims, ann))
        iv.append((k["ts"], k["ts"] + k["dur"]))
    iv.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (iv[-1][1] - iv[0][0]) if iv else 0.0
    return {"by_path": dict(by_path), "kernels": rows, "busy_us": busy, "span_us": span, "n_kernels": len(kernels)}


def kernel_class(name):
    """Category of a kernel by its name (for compiled runs and CUDA-graph replays)."""
    n = name.lower()
    if "_bmm_kernel" in n:
        return "attention (bmm q.k / p.v)"
    if "_max_kernel" in n or "_acc_kernel" in n:
        return "attention (fused paged)"
    if ("gemm" in n or "cutlass" in n or "xmma" in n or "imma" in n or "s8" in n and "mm" in n) and "triton" not in n:
        return "int GEMM (cuBLASLt _int_mm)"
    if n.startswith("triton_"):
        return "fused elementwise (Inductor)"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "eager elementwise"
    if "reduce" in n:
        return "eager reduction"
    if "index" in n or "gather" in n or "scatter" in n or "cat" in n:
        return "index/copy"
    if "memcpy" in n or "memset" in n or "fill" in n:
        return "memcpy/fill"
    return "other"
