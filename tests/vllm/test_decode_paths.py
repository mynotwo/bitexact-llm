"""Decode paths against the integer reference on the real constants (needs CUDA and BITEXACT_INT8_CONST).

Run: BITEXACT_INT8_CONST=int8_const.pt pytest -q tests/vllm/test_decode_paths.py

Two prompts are prefilled with the plugin's eager paged path, then decoded 6 steps two ways. Each step's logits
are compared with bitexact.int8's IntLlama (its own cache, eager): (1) graph.decode_forward eager, (2) the same
under a CUDA graph (what DecodeGraphs and vLLM replay). This catches compiled regions that lose the KV write and
capture-time host syncs.
"""

import os
import types

import pytest
import torch

CONST = os.environ.get("BITEXACT_INT8_CONST")
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not (torch.cuda.is_available() and CONST), reason="needs CUDA + BITEXACT_INT8_CONST")]


def _run(use_graph):
    from bitexact.int8.model import IntLlama
    from bitexact.llama import Cache
    from bitexact.vllm_plugin import paged, tri_paged
    from bitexact.vllm_plugin.graph import decode_forward
    m = IntLlama(torch.load(CONST, map_location="cpu", weights_only=False), "cuda", nl=2, compiled=True)
    m.ptab = tri_paged.weight_table(m.exp)
    dev, bs, R, steps = m.dev, 16, 2, 6
    lens = [37, 150]
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(1000, 100000, (n,), generator=g).tolist() for n in lens]
    nblk = (max(lens) + steps + bs - 1) // bs
    kvs = [torch.zeros(R * nblk + 1, 8, bs, 256, dtype=torch.bfloat16, device=dev) for _ in range(m.L)]
    bt = (torch.arange(R * nblk, device=dev) + 1).view(R, nblk)
    # plugin prefill (eager paged path), both prompts in one batch
    ids = torch.tensor(sum(prompts, []), device=dev)
    pos = torch.cat([torch.arange(n) for n in lens]).to(dev)
    qsl = torch.tensor([0, lens[0], sum(lens)])
    slot = torch.cat([bt[r, torch.arange(n, device=dev) // bs] * bs + torch.arange(n, device=dev) % bs
                      for r, n in enumerate(lens)])
    meta = types.SimpleNamespace(query_start_loc=qsl, seq_lens=torch.tensor(lens), block_table=bt,
                                 slot_mapping=slot)
    plan = paged.make_plan(meta, len(ids), pos)
    x = m.emb[ids].long()
    for i in range(m.L):
        x = paged.attention(m, i, x, pos, kvs[i], plan)
        x = paged.mlp(m, m.layers[i], x)
    last = x[torch.tensor([lens[0] - 1, sum(lens) - 1], device=dev)]
    # integer reference: bitexact.int8 on its own cache
    refs = []
    for p in prompts:
        c = Cache(m.L, 1, len(p) + steps, dev)
        refs.append((c, m.forward(torch.tensor([p], device=dev), torch.arange(len(p), device=dev)[None], c)[:, -1]))
    li = m.logits(last)
    tok = torch.zeros(R, dtype=torch.long, device=dev)
    p_t = torch.tensor(lens, device=dev)
    s_t = torch.zeros(R, dtype=torch.long, device=dev)
    graph = None
    for t in range(steps):
        for r in range(R):
            assert torch.equal(li[r], m.logits(refs[r][1])[0]), (use_graph, t, r)
        tok.copy_(li.argmax(-1))
        s_t.copy_(bt[torch.arange(R, device=dev), p_t // bs] * bs + p_t % bs)
        fwd = lambda: decode_forward(m, kvs, tok, p_t, s_t, bt)              # noqa: E731
        if use_graph:
            if graph is None:
                fwd()                                                       # warm-up (compiles)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = fwd()
            graph.replay()
            h = out
        else:
            h = fwd()
        li = m.logits(h)
        for r in range(R):
            c, _ = refs[r]
            refs[r] = (c, m.forward(tok[r].view(1, 1), p_t[r].view(1, 1), c)[:, -1])
        p_t += 1
    m.check()


def test_decode_eager():
    _run(False)


def test_decode_cuda_graph():
    _run(True)
