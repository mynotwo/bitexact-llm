"""Throughput of the non-vLLM integer engines (exact or int8 mode) on the vLLM plugin's workload.

B random prompts of --in-len tokens (the same prompts as tests/vllm/bench.py: random.Random(0),
ids in [1000, 120000)), --out-len greedy tokens each (no EOS stop). One pass = one batched prefill
forward, then out_len - 1 batched decode steps through the integer KV cache. The same pass runs twice;
the second is timed (output tokens / seconds, prefill included), as in the plugin numbers.

  python bench_engine.py --const exact_const.pt --batch 32 [--graph] [--compiled] --out r.json

--graph: every decode step is one CUDA graph replay at fixed key length S = in_len + out_len (keys beyond
a query's position are masked, so the integers do not change); captured before pass 1, reused in pass 2.
Without --graph the decode step runs eagerly with S = current length.
Checks: pass 1 and pass 2 give the same token ids; rows listed in --ref-rows are compared with
bitexact.generate (one prompt alone, eager, its own cache); model.check() after all passes.
Token ids are saved to --tokens (torch [B, out_len] int64) for compare_tokens.py.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch


class RowCache:
    """Rows [r0, r1) of a Cache, as a cache of its own (views: writes land in the full cache)."""

    def __init__(self, cache, r0, r1):
        self.k = [t[r0:r1] for t in cache.k]
        self.v = [t[r0:r1] for t in cache.v]
        self.write = type(cache).write.__get__(self)


def prompts(B, n, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(1000, 120000) for _ in range(n)] for _ in range(B)]


class Engine:
    def __init__(self, model, B, P, N, graph, prefill_rows):
        from bitexact.intops import first_argmax
        self.m, self.B, self.P, self.N = model, B, P, N
        self.argmax = first_argmax
        self.rows = prefill_rows or B
        self.cache = model.new_cache(B, P + N)
        self.graph = None
        if graph:
            self._capture()

    def _step_body(self, y, pos, S):
        m = self.m
        x = m.forward(y.view(-1, 1), pos, self.cache, S)
        return self.argmax(m.logits(x[:, 0]))

    def _graph_body(self, y, pos, S):
        """The exact model rebinds m.err to new tensors; the graph must write into a flag tensor that stays
        alive (self.serr, held here), or a replay writes into freed memory. serr accumulates over replays."""
        m = self.m
        m.err = self.serr
        out = self._step_body(y, pos, S)
        if m.err is not self.serr:
            self.serr.copy_(m.err)
        m.err = self.serr.clone()
        return out

    def _capture(self):
        dev, B = self.m.dev, self.B
        self.serr = torch.zeros((), dtype=torch.bool, device=dev)
        self.sy = torch.zeros(B, dtype=torch.long, device=dev)
        self.spos = torch.full((B, 1), self.P, dtype=torch.long, device=dev)
        S = self.P + self.N
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                 # warm up (torch.compile, Triton autotune, allocator)
            for _ in range(3):
                self._graph_body(self.sy, self.spos, S)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.sout = self._graph_body(self.sy, self.spos, S)
        torch.cuda.synchronize()
        self.m.err = self.serr.clone()

    def step(self, y, p):
        """Tokens y [B] at position p -> next tokens [B]."""
        if self.graph is not None:
            self.sy.copy_(y)
            self.spos.fill_(p)
            self.graph.replay()
            return self.sout.clone()
        pos = torch.full((self.B, 1), p, dtype=torch.long, device=self.m.dev)
        return self._step_body(y, pos, p + 1)

    @torch.no_grad()
    def run(self, ids):
        """ids [B, P] on device -> (tokens [B, N], total s, prefill s)."""
        m, B, P = self.m, self.B, self.P
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pos = torch.arange(P, device=m.dev).expand(B, P).contiguous()
        firsts = []
        for r0 in range(0, B, self.rows):
            r1 = min(B, r0 + self.rows)
            c = self.cache if (r0, r1) == (0, B) else RowCache(self.cache, r0, r1)
            x = m.forward(ids[r0:r1], pos[r0:r1], c, P)
            firsts.append(self.argmax(m.logits(x[:, -1])))
        y = torch.cat(firsts)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        toks = [y]
        for t in range(1, self.N):
            y = self.step(y, P + t - 1)
            toks.append(y)
        out = torch.stack(toks, 1)
        if self.graph is not None:
            m.err = m.err | self.serr
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0, t1 - t0


def tok_sha(t):
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--gemm", default="triton", help="exact mode only: triton or cublas")
    ap.add_argument("--n-layers", type=int, default=None, help="first N layers only (smoke)")
    ap.add_argument("--prefill-rows", type=int, default=0, help="prompts per prefill forward (0: all)")
    ap.add_argument("--ref-rows", default="0,-1", help="rows checked against bitexact.generate ('' = none)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import bitexact
    from bitexact import generate, load
    print(f"bitexact from {bitexact.__file__}", flush=True)
    t = time.time()
    c = torch.load(a.const, map_location="cpu", weights_only=False, mmap=True)
    mode = "exact" if "lm.limbs" in c else "int8"
    kw = {"n_layers": a.n_layers, "compiled": a.compiled}
    if mode == "exact":
        kw["gemm"] = a.gemm
    model = load(c, "cuda", **kw)
    sha = c.get("sha256")
    del c
    load_s = time.time() - t
    print(f"loaded {mode} in {load_s:.1f}s sha={str(sha)[:16]} layers={model.L}", flush=True)

    B, P, N = a.batch, a.in_len, a.out_len
    ps = prompts(B, P)
    ids = torch.tensor(ps, device=model.dev)
    t = time.time()
    eng = Engine(model, B, P, N, a.graph, a.prefill_rows)
    cap_s = time.time() - t
    tok1, sec1, pre1 = eng.run(ids)
    tok2, sec2, pre2 = eng.run(ids)
    model.check()
    same = bool(torch.equal(tok1, tok2))
    ref = {}
    for r in [int(x) for x in a.ref_rows.split(",") if x.strip()]:
        r = r % B
        if r in ref:
            continue
        rt, _ = generate(model, ps[r], N)
        ref[r] = rt == tok2[r].tolist()
    n = B * N
    res = {"tag": a.tag, "mode": mode, "batch": B, "graph": a.graph, "compiled": a.compiled,
           "gemm": a.gemm if mode == "exact" else "int_mm", "layers": model.L, "in_len": P, "out_len": N,
           "out_tokens": n, "sec": round(sec2, 4), "tok_per_s": round(n / sec2, 1),
           "prefill_s": round(pre2, 4), "decode_tok_per_s": round(B * (N - 1) / (sec2 - pre2), 1),
           "pass1_sec": round(sec1, 3), "capture_s": round(cap_s, 1), "load_s": round(load_s, 1),
           "pass1_eq_pass2": same, "ref_rows_match": {str(k): v for k, v in ref.items()},
           "tokens_sha16": tok_sha(tok2), "const_sha16": str(sha)[:16],
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    if a.tokens:
        torch.save(tok2.cpu(), a.tokens)
    line = json.dumps(res)
    if a.out:
        with open(a.out, "w") as f:
            f.write(line + "\n")
    print("RESULT " + line, flush=True)
    if not same or not all(ref.values()):
        print("ERROR: token ids differ between passes or from bitexact.generate", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
    main()
