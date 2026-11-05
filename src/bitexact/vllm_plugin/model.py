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
itself is chosen from the exact integers by sampler.IntSampler (the greedy one computed in the same pass as
the float copy, tri_logits.py).
"""

import os

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention

from bitexact.exact.consts import FT as EXACT_FT
from bitexact.int8.consts import FT
from bitexact.int8.model import IntLlama
from bitexact.llama import rope_tables
from bitexact.vllm_plugin import paged, prefill, tri_logits, tri_paged, tri_sample
from bitexact.vllm_plugin.graph import BUCKETS
from bitexact.vllm_plugin.sampler import STASH

MODEL = None             # the loaded model in this process (the sampler reads the ln table from it)


def _check_on_output():
    """Wraps vLLM's AsyncOutput.get_output (V2 model runner) to raise on the bound-violation flag. get_output
    waits for the step's output copy, which is ordered after compute_logits' flag copy on the main stream, so
    the flag covers that step without another sync (m.err is never cleared: a later copy only sets it sooner)."""
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    get = AsyncOutput.get_output
    if getattr(get, "bitexact", False):
        return

    def get_output(self):
        out = get(self)
        if MODEL is not None and bool(MODEL.flag):
            raise OverflowError("integer bound violated (int8/model.py or exact)")
        return out
    get_output.bitexact = True
    AsyncOutput.get_output = get_output


def _gl_hook():
    """Wraps vLLM's FULL-graph replay: the step's replay carries the LM-head pass (see _glogits); here the host
    picks, from the running requests, whether that pass is the one this step needs (greedy rows, nothing else)
    and hands its outputs to compute_logits."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    run = ModelCudaGraphManager.run_fullgraph
    if getattr(run, "bitexact", False):
        return

    def run_fullgraph(self, desc):
        if MODEL is not None:
            MODEL._gl_replay(desc.num_tokens)
        return run(self, desc)
    run_fullgraph.bitexact = True
    ModelCudaGraphManager.run_fullgraph = run_fullgraph


def _extend_rope(c, model_dir, n):
    """The constants hold 2048 rows of integer cos / sin. For longer contexts rebuild n rows with the same
    rule (llama3 scaling, float64, rounded once) and check the first rows are bit-identical."""
    if c["cos"].shape[0] >= n:
        return
    ft = EXACT_FT if "lm.limbs" in c else FT       # exact mode keeps cos / sin at 30 bits
    with torch.device("cpu"):                 # vLLM sets a CUDA default device while it builds models
        cos, sin = rope_tables(model_dir, n, ft)
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
        compiled = bool(int(os.environ.get("BITEXACT_COMPILE", "1")))
        self.exact = "lm.limbs" in c                    # constants of exact mode (bf16 weights) or int8 mode
        if self.exact:
            from bitexact.vllm_plugin.exact_run import ExactEngine
            self.m = ExactEngine(c, dev, compiled=compiled)
        else:
            self.m = IntLlama(c, dev, nl=int(getattr(cfg, "bitexact_nl", 2)), compiled=compiled)
            self.m.ptab = tri_paged.weight_table(self.m.exp)
        # Plugin-side graphs only when vLLM's own are off (enforce_eager); otherwise vLLM captures
        # uniform decode batches through _captured_decode.
        self.use_graphs = bool(int(os.environ.get("BITEXACT_GRAPHS", "1"))) and \
            vllm_config.model_config.enforce_eager
        self.max_len = vllm_config.model_config.max_model_len
        sc = vllm_config.speculative_config
        self.spec_q = 1 + (sc.num_speculative_tokens if sc is not None else 0)    # query tokens per request
        self.graphs = None
        self.sha = c["sha256"]
        log = os.environ.get("BITEXACT_HASH_LOG")
        self.plan_log = log and log[:-len(".hash")] + ".plan"    # batch layout per forward (plan_stats.py)
        # host copy of the bound-violation flag, read as each step's output reaches the host (_check_on_output)
        self.flag = torch.zeros((), dtype=torch.bool, device="cpu", pin_memory=torch.cuda.is_available())
        del c
        # Attention layers only to get vLLM's KV cache: HKV heads x 2 HD bf16 per token (= K | V as 2 HD int32,
        # 8 HD bytes: 3 int8 limb planes each, tri_paged.py).
        m = self.m
        self.qdim = m.H * m.HD
        self.attn = nn.ModuleList([
            Attention(num_heads=m.HKV, head_size=2 * m.HD, scale=1.0, num_kv_heads=m.HKV,
                      cache_config=vllm_config.cache_config, prefix=f"model.layers.{i}.self_attn.attn")
            for i in range(self.m.L)])
        self.layer0 = "model.layers.0.self_attn.attn"
        if os.environ.get("BITEXACT_COMPILE", "1") == "1":
            # Inductor initialises its pattern tables on the first compile (host-to-device copies); do it now,
            # never inside a CUDA graph capture (speculative decoding reaches the first compile there).
            torch.compile(lambda t: t * 2 + 1)(torch.zeros(8, dtype=torch.long, device=dev))
        # LM head inside the decode CUDA graphs (batches up to graph.BUCKETS[-1] rows; not with speculative decoding)
        self.gl = os.environ.get("BITEXACT_GRAPH_LOGITS", "1") == "1" and self.spec_q == 1
        self.gl_side, self.gl_warm, self.gl_val = {}, set(), 0
        self.gate = torch.zeros(1, dtype=torch.int32, device=dev)
        STASH.active = True
        MODEL = self
        _check_on_output()
        if self.gl:
            _gl_hook()

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
        R = bt.shape[0]
        if self.exact:                                     # one table row per request; rmap = row of each token
            rmap = torch.arange(N, device=bt.device) // self.spec_q
            return self.m.forward([a.kv_cache for a in self.attn], input_ids, positions, meta.slot_mapping[:N], bt,
                                  rmap.clamp(max=R - 1))
        if R != N and self.spec_q > 1:
            # speculative decoding: uniform q = N / R query tokens per request (verify rows), each row attends
            # with its own position (causal) through its request's block table
            q = self.spec_q
            bt = bt[(torch.arange(N, device=bt.device) // q).clamp(max=R - 1)]
        elif R != N:
            bt = bt[torch.arange(N, device=bt.device).clamp(max=R - 1)]
        x = decode_forward(self.m, [a.kv_cache for a in self.attn], input_ids, positions, meta.slot_mapping[:N], bt)
        if self.gl and N <= BUCKETS[-1]:
            self._glogits(x, N)
        return x

    def _glogits(self, x, N):
        """The LM head after the decode forward, as part of its CUDA graph: final norm, LM-head GEMM and, behind a
        device gate, the one-hot / greedy pass. The host then has no eager launches for a greedy step (the gate is
        set per replay in _gl_replay); the GEMM output stays for the other modes (noisy, float) that run eagerly
        on it in compute_logits. Same kernels and arguments as the eager path. A first call outside the capture
        (vLLM's warm-up) compiles them."""
        mm = self.m
        cap = torch.cuda.is_current_stream_capturing()
        if mm.nl != 2 or mm.post_check or not x.is_cuda or (not cap and N in self.gl_warm):
            return
        pre = mm.sg.pre_norm(x.long(), mm.gf)
        w, mw, fmsh = mm.lm
        head = tri_sample.Head(mm._gemm(pre[0], w), mw, fmsh, pre[1], pre[2])
        out, y = tri_sample.head(head, "onehot", gate=self.gate)
        if cap:
            self.gl_side[N] = (head, out, y)
        else:
            self.gl_warm.add(N)

    def _gl_replay(self, N):
        """Called right before vLLM replays the graph of N tokens: decide the step's logits mode from the running
        requests (the same test compute_logits applies on the eager path) and set the gate."""
        STASH.gl = self.gl_side.get(N)
        if STASH.gl is None:
            return
        sp = STASH.sampler
        strict = sp is None or sp.any_strict()
        noisy = (not strict) and sp.any_noisy()
        STASH.gmode = (strict, noisy)
        v = int(not strict and not noisy)
        if v != self.gl_val:
            self.gate.fill_(v)
            self.gl_val = v

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
        meta = self._meta()
        STASH.gl = None
        if meta is not None and torch.cuda.is_current_stream_capturing():
            return self._captured_decode(input_ids, positions, meta)
        if self.exact:
            return self._forward_exact(input_ids, positions, meta)
        x = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds.long()
        N = x.shape[0]
        plan = None if meta is None else paged.make_plan(meta, N, positions, self.qdim)
        if plan is not None and self.plan_log:
            self._log_plan(plan)
        if plan is not None and not self.use_graphs and plan.n == N and (
                (not plan.pre and plan.dec_req.numel() == N) or
                (self.spec_q > 1 and not plan.dec_req.numel() and all(ql == self.spec_q for _, _, ql, _ in plan.pre))):
            # Uniform decode with vLLM's graphs on: the device-only path vLLM captures. vLLM's warm-up run
            # calls it right before each capture, so the kernels are compiled and loaded outside the capture.
            return self._captured_decode(input_ids, positions, meta)
        if self.use_graphs and plan is not None:
            if self.graphs is None:
                from bitexact.vllm_plugin.graph import DecodeGraphs
                self.graphs = DecodeGraphs(self.m, [a.kv_cache for a in self.attn], self.max_len)
            if self.graphs.applies(plan, N):
                return self.graphs.run(input_ids, positions.long(), plan)
        if inputs_embeds is None and plan is not None and prefill.applies(self.m, plan, N):
            return prefill.forward(self.m, [a.kv_cache for a in self.attn], input_ids, positions, plan)
        for i in range(self.m.L):
            x = paged.attention(self.m, i, x, positions, self.attn[i].kv_cache, plan)
            x = paged.mlp(self.m, self.m.layers[i], x)
        self.m._flag(x.abs().amax() >= 1 << 42)
        return x

    def _forward_exact(self, input_ids, positions, meta):
        """Any mix of decode and prefill tokens: every token is a row with its request's block-table row."""
        kvs = None if meta is None else [a.kv_cache for a in self.attn]
        N = input_ids.shape[0]
        if meta is None:
            return self.m.forward(None, input_ids, positions, input_ids, input_ids, input_ids)
        qsl = meta.query_start_loc.cpu().tolist()
        R = len(meta.seq_lens)
        n = min(qsl[R], N)
        if self.plan_log:                                   # same line as _log_plan (plan_stats.py)
            sl = meta.seq_lens.cpu().tolist()
            ql = [qsl[r + 1] - qsl[r] for r in range(R)]
            with open(self.plan_log, "a") as f:
                f.write(f"{n} {sum(q == 1 for q in ql)} {sum(q > 1 for q in ql)} "
                        f"{sum(q > 1 and sl[r] - q > 0 for r, q in enumerate(ql))}\n")
        if n == N == R and self.spec_q == 1 and qsl[R] == R:
            # uniform decode: the device-only path vLLM captures; its warm-up run before each capture comes
            # through here, so the compiled stages are built outside the capture
            return self._captured_decode(input_ids, positions, meta)
        rmap = torch.zeros(N, dtype=torch.long)
        rmap[:n] = torch.repeat_interleave(torch.arange(R), torch.diff(torch.tensor(qsl[:R + 1])))[:n]
        slot = torch.full((N,), -1, dtype=torch.long, device=input_ids.device)
        slot[:n] = meta.slot_mapping[:n]
        pos = positions.clone()
        pos[n:] = 0
        return self.m.forward(kvs, input_ids, pos, slot, meta.block_table, rmap.to(input_ids.device))

    def _check_late(self):
        """The bound-violation flag without a host sync: m.err (sticky, ORed on the device) is copied to pinned
        memory each step and read in that step's AsyncOutput.get_output (_check_on_output), after vLLM's own
        wait for the step's outputs. So a violation raises before the step's tokens are returned, the last
        step included. BITEXACT_HASH_LOG (test runs) keeps the immediate check."""
        if self.plan_log:
            return self.m.check()
        self.flag.copy_(self.m.err, non_blocking=True)

    def compute_logits(self, hidden_states):
        mm = self.m
        if self.exact:
            li = mm.logits(hidden_states.long())
            self._check_late()
            STASH.head, STASH.logits = None, li
            STASH.strict = STASH.sampler is None or STASH.sampler.any_strict()
            # strict requests need vLLM's real float logits; otherwise the sampler rewrites every row, so only
            # the greedy token is computed (no float copy of the [rows, V] logits)
            f, STASH.greedy = tri_logits.argmax_float(li, scale=2.0 ** -mm.FR, write=STASH.strict)
            return f if STASH.strict else torch.empty(li.shape, dtype=torch.float32, device=li.device)
        if hidden_states.is_cuda and mm.nl == 2 and not mm.post_check:
            # LM-head GEMM, then one Triton pass over its output (tri_sample): no int64 logits are stored. A decode
            # step replayed from a CUDA graph arrives with the GEMM (and the greedy pass) already done (_glogits).
            g, STASH.gl = STASH.gl, None
            R = hidden_states.shape[0]
            if g is not None and R <= g[0].M:
                h = g[0]
                head = tri_sample.Head(h.c, h.m, h.fmsh, h.mant, h.e, R)
                strict, noisy = STASH.gmode
                if os.environ.get("BITEXACT_GL_CHECK") == "1":
                    self._gl_check(hidden_states, head, g[2][:R])
            else:
                g = None
                pre = mm.sg.pre_norm(hidden_states.long(), mm.gf)
                w, mw, fmsh = mm.lm
                head = tri_sample.Head(mm._gemm(pre[0], w), mw, fmsh, pre[1], pre[2])
                strict = STASH.sampler is None or STASH.sampler.any_strict()
                noisy = (not strict) and STASH.sampler.any_noisy()
            self._check_late()
            STASH.head, STASH.logits = head, None
            STASH.strict = strict
            if not strict and noisy:
                # temperature rows: the sampler's noisy pass chooses every token and rewrites all of vLLM's
                # logits, so no pass here (and no argmax) and the buffer need not be initialised.
                STASH.greedy = None
                return torch.empty(head.R, head.V, dtype=torch.float32, device=head.c.device)
            if g is not None and not strict:
                STASH.greedy = g[2][:R]
                return g[1][:R]
            f, STASH.greedy = tri_sample.head(head, "float" if strict else "onehot")
            return f
        li = mm.logits(hidden_states.long())
        self._check_late()
        STASH.head, STASH.logits = None, li
        if not li.is_cuda:
            STASH.greedy = None
            return li.float() * (2.0 ** -16)
        f, STASH.greedy = tri_logits.argmax_float(li)          # same float bits, plus the greedy token
        return f

    def _gl_check(self, hidden_states, head, y):
        """Test mode (BITEXACT_GL_CHECK=1): the graph's GEMM output and greedy tokens equal a recomputation from
        the hidden states vLLM handed over (same rows, same integers)."""
        from bitexact.intops import join_limbs
        mm = self.m
        pre = mm.sg.pre_norm(hidden_states.long(), mm.gf)
        c = mm._gemm(pre[0], mm.lm[0])
        R = head.R
        assert torch.equal(pre[1], head.mant[:R]) and torch.equal(pre[2], head.e[:R]), "graph logits: norm differs"
        assert torch.equal(join_limbs(c, R, 2), join_limbs(head.c, head.M, 2)[:R]), "graph logits: GEMM differs"
        if self.gate.item():
            _, y2 = tri_sample.head(tri_sample.Head(c, *mm.lm[1:], pre[1], pre[2]), "none")
            assert torch.equal(y, y2), "graph logits: greedy tokens differ"

    def load_weights(self, weights):
        for _ in weights:            # the integer constants were loaded in __init__
            pass
        return set()
