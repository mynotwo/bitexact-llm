"""vLLM model class for the int8 engine (Llama-3.2-1B).

The forward is bitexact.int8 (IntLlama: int8 weights, int8 limb GEMMs with int32 accumulation, integer
RMSNorm / RoPE / softmax / SiLU) on vLLM's flattened token batch, with attention on vLLM's paged KV cache
(paged.py). Weights are not read from the checkpoint: the integer constants file (built once by
bitexact.int8.consts) is loaded from config.json's "bitexact_const" (or $BITEXACT_CONST).

Each token row is computed with row-local integer rules and exact integer reductions, so the hidden
state and the logits of a token are the same integers whatever else vLLM batches with it.

Speed: the elementwise integer segments run under torch.compile (BITEXACT_COMPILE=1, default).
Decode-only steps run as CUDA graphs: vLLM's own (default; FULL_DECODE_ONLY, captured through
_captured_decode) or, with enforce_eager=True, the plugin's (graph.py, BITEXACT_GRAPHS=1).

Hidden states returned to vLLM are int64 at FR = 16 fractional bits. compute_logits returns the
integer logits / 2^16 as float32 for vLLM's bookkeeping (logprobs are therefore approximate); the token
itself is chosen from the exact integers by sampler.IntSampler.
"""

import os

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention

from bitexact.int8.consts import FT
from bitexact.int8.model import IntLlama
from bitexact.llama import rope_tables
from bitexact.vllm_plugin import paged, tri_paged
from bitexact.vllm_plugin.sampler import STASH

MODEL = None             # the loaded model in this process (the sampler reads the ln table from it)


def _extend_rope(c, model_dir, n):
    """The constants hold 2048 rows of integer cos / sin. For longer contexts rebuild n rows with the same
    rule (llama3 scaling, float64, rounded once) and check the first rows are bit-identical."""
    if c["cos"].shape[0] >= n:
        return
    with torch.device("cpu"):                 # vLLM sets a CUDA default device while it builds models
        cos, sin = rope_tables(model_dir, n, FT)
    m = c["cos"].shape[0]
    assert torch.equal(cos[:m], c["cos"]) and torch.equal(sin[:m], c["sin"]), "RoPE rows differ"
    c["cos"], c["sin"] = cos, sin


class BitExactLlamaForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        global MODEL
        cfg = vllm_config.model_config.hf_config
        path = os.environ.get("BITEXACT_CONST") or getattr(cfg, "bitexact_const", None)
        assert path, "set bitexact_const in config.json or $BITEXACT_CONST"
        c = torch.load(path, map_location="cpu", weights_only=False)
        _extend_rope(c, vllm_config.model_config.model, vllm_config.model_config.max_model_len)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.m = IntLlama(c, dev, nl=int(getattr(cfg, "bitexact_nl", 2)),
                          compiled=bool(int(os.environ.get("BITEXACT_COMPILE", "1"))))
        self.m.ptab = tri_paged.weight_table(self.m.exp)
        # Plugin-side graphs only when vLLM's own are off (enforce_eager); otherwise vLLM captures
        # uniform decode batches through _captured_decode.
        self.use_graphs = bool(int(os.environ.get("BITEXACT_GRAPHS", "1"))) and \
            vllm_config.model_config.enforce_eager
        self.max_len = vllm_config.model_config.max_model_len
        self.graphs = None
        self.sha = c["sha256"]
        log = os.environ.get("BITEXACT_HASH_LOG")
        self.plan_log = log and log[:-len(".hash")] + ".plan"    # batch layout per forward (plan_stats.py)
        self.pending = []        # (event, pinned flag) of overflow checks not yet read (_check_late)
        del c
        # Attention layers only to get vLLM's KV cache: 8 KV heads x 128 bf16 = 8 x 64 int32 (K | V).
        self.attn = nn.ModuleList([
            Attention(num_heads=8, head_size=128, scale=1.0, num_kv_heads=8,
                      cache_config=vllm_config.cache_config, prefix=f"model.layers.{i}.self_attn.attn")
            for i in range(self.m.L)])
        self.layer0 = "model.layers.0.self_attn.attn"
        STASH.active = True
        MODEL = self

    def embed_input_ids(self, input_ids):
        return self.m.emb[input_ids].long()

    def _meta(self):
        meta = get_forward_context().attn_metadata
        return meta.get(self.layer0) if isinstance(meta, dict) else meta

    def _log_plan(self, plan):
        """One line per forward: tokens, decode requests, prefill requests, and of those how many start
        after position 0 (a later chunk of a chunked prefill, or a prefix-cache hit)."""
        with open(self.plan_log, "a") as f:
            f.write(f"{plan.n} {plan.dec_req.numel()} {len(plan.pre)} "
                    f"{sum(sl - ql > 0 for _, _, ql, sl in plan.pre)}\n")

    def _captured_decode(self, input_ids, positions, meta):
        """Uniform decode batch (one token per request, batch order = block-table rows), device-only forward;
        this is what vLLM captures and replays."""
        from bitexact.vllm_plugin.graph import decode_forward
        N = input_ids.shape[0]
        bt = meta.block_table
        if bt.shape[0] != N:
            bt = bt[torch.arange(N, device=bt.device).clamp(max=bt.shape[0] - 1)]
        return decode_forward(self.m, [a.kv_cache for a in self.attn], input_ids, positions,
                              meta.slot_mapping[:N], bt)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        meta = self._meta()
        if meta is not None and torch.cuda.is_current_stream_capturing():
            return self._captured_decode(input_ids, positions, meta)
        x = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds.long()
        N = x.shape[0]
        plan = None if meta is None else paged.make_plan(meta, N, positions)
        if plan is not None and self.plan_log:
            self._log_plan(plan)
        if plan is not None and not self.use_graphs and not plan.pre and plan.n == N \
                and plan.dec_req.numel() == N:
            # Uniform decode with vLLM's graphs on: the device-only path vLLM captures. vLLM's warm-up run
            # calls it right before each capture, so the kernels are compiled and loaded outside the capture.
            return self._captured_decode(input_ids, positions, meta)
        if self.use_graphs and plan is not None:
            if self.graphs is None:
                from bitexact.vllm_plugin.graph import DecodeGraphs
                self.graphs = DecodeGraphs(self.m, [a.kv_cache for a in self.attn], self.max_len)
            if self.graphs.applies(plan, N):
                return self.graphs.run(input_ids, positions.long(), plan)
        for i in range(self.m.L):
            x = paged.attention(self.m, i, x, positions, self.attn[i].kv_cache, plan)
            x = paged.mlp(self.m, self.m.layers[i], x)
        self.m._flag(x.abs().amax() >= 1 << 42)
        return x

    def _check_late(self):
        """The bound-violation flag without a host sync: its value is copied to pinned memory each step and
        read once the copy is done (usually at the next step). A violation raises a step or two late; while
        no bound is violated nothing changes. BITEXACT_HASH_LOG (test runs) keeps the immediate check."""
        if self.plan_log:
            return self.m.check()
        while self.pending and self.pending[0][0].query():
            if bool(self.pending.pop(0)[1]):
                raise OverflowError("integer bound violated (int8/model.py)")
        flag = torch.empty((), dtype=torch.bool, pin_memory=True)
        flag.copy_(self.m.err, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.pending.append((ev, flag))

    def compute_logits(self, hidden_states):
        li = self.m.logits(hidden_states.long())
        self._check_late()
        STASH.logits = li
        return li.float() * (2.0 ** -16)

    def load_weights(self, weights):
        for _ in weights:            # the integer constants were loaded in __init__
            pass
        return set()
