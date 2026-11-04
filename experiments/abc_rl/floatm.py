"""Float side of the A/B/C RL experiment: vLLM bf16 sampler with per-step weight sync, and an HF bf16
trainer forward over the dequantized weights with a straight-through gradient to the fp32 master.

Both float engines receive the same tensors: the dequantized BitExact int8 weights (intq.quantize), stored
in bf16 (one rounding, relative error <= 2^-9, as any bf16 stack stores its weights).
"""

import json
import os

import torch
import torch.nn.functional as F


def write_model_dir(src, fw, out):
    """HF model dir with untied lm_head holding the dequantized initial weights (vLLM needs a full set)."""
    from safetensors.torch import save_file
    if os.path.exists(os.path.join(out, "model.safetensors")):
        return out
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["tie_word_embeddings"] = False
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in os.listdir(src):
        if f not in ("config.json", "model.safetensors") and not os.path.exists(os.path.join(out, f)):
            os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(out, f))
    tmp = os.path.join(out, "model.safetensors.tmp")
    save_file({k: v.to(torch.bfloat16).contiguous().cpu() for k, v in fw.items()}, tmp)
    os.replace(tmp, os.path.join(out, "model.safetensors"))
    return out


class VllmSampler:
    """In-process vLLM (V1 multiprocessing off) so weights can be loaded into the live model every step."""

    def __init__(self, model_dir, gpu_mem, max_seqs=512):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        self.llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=1024, max_num_seqs=max_seqs,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0)

    def load(self, fw):
        ws = [(k, v.to(torch.bfloat16)) for k, v in fw.items()]

        def _load(worker):
            mr = worker.model_runner
            model = mr.get_model() if hasattr(mr, "get_model") else mr.model
            model.load_weights(iter(ws))
            return True
        self.llm.collective_rpc(_load)
        self.llm.reset_prefix_cache()

    def generate(self, prompts, seeds, max_new, greedy=False):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = [SamplingParams(temperature=0.0 if greedy else 1.0, top_p=1.0, seed=s, max_tokens=max_new,
                             logprobs=None if greedy else 0) for s in seeds]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        toks, lps, texts = [], [], []
        for o in outs:
            c = o.outputs[0]
            t = list(c.token_ids)
            toks.append(t)
            texts.append(c.text)
            lps.append(None if greedy else [c.logprobs[i][x].logprob for i, x in enumerate(t)])
        return toks, lps, texts


class FloatTrainer:
    """HF LlamaForCausalLM (sdpa, untied) run through functional_call on bf16 (or fp32) casts of fp32 leaves
    that hold the dequantized weights. Gradients accumulate on the leaves and are copied to the master
    (straight-through estimator: d quant(P) / dP := I)."""

    def __init__(self, model_dir, device, dtype=torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_dir)
        cfg.tie_word_embeddings = False
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to(device)
        self.model.eval()                       # no dropout in Llama; eval only fixes flags
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dev = torch.device(device)
        self.dtype = dtype
        self.leaves = None

    def set_weights(self, fw):
        if self.leaves is None:
            self.leaves = {k: torch.empty_like(v, device=self.dev).requires_grad_(True) for k, v in fw.items()}
        with torch.no_grad():
            for k, v in fw.items():
                self.leaves[k].copy_(v)
                self.leaves[k].grad = None

    def _hidden(self, ids, mask, params):
        from torch.func import functional_call
        out = functional_call(self.model.model, {k[len("model."):]: v for k, v in params.items()
                                                 if k.startswith("model.")},
                              args=(), kwargs=dict(input_ids=ids, attention_mask=mask))
        return out.last_hidden_state

    def logp(self, prompts, resps, grad):
        """Per-token float log p of resps given prompts (one microbatch). Returns a flat tensor over the
        concatenated response tokens (with grad if requested) and the split sizes."""
        B = len(prompts)
        L = max(len(p) + len(r) for p, r in zip(prompts, resps))
        ids = torch.zeros((B, L), dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, (p, r) in enumerate(zip(prompts, resps)):
            ids[i, :len(p) + len(r)] = torch.tensor(p + r)
            mask[i, :len(p) + len(r)] = 1
        ids, mask = ids.to(self.dev), mask.to(self.dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            params = {k: v.to(self.dtype) for k, v in self.leaves.items()}
            h = self._hidden(ids, mask, params)
            sel = torch.cat([h[i, len(p) - 1:len(p) - 1 + len(r)] for i, (p, r) in enumerate(zip(prompts, resps))])
            tok = torch.tensor([t for r in resps for t in r], device=self.dev)
            out = []
            for k in range(0, sel.shape[0], 2048):          # bound the [n, V] float32 buffer
                lg = F.linear(sel[k:k + 2048], params["lm_head.weight"]).float()
                out.append(lg.log_softmax(-1).gather(1, tok[k:k + 2048, None])[:, 0])
            return torch.cat(out), [len(r) for r in resps]

    def grads_to_master(self, P):
        """Straight-through: master grad = leaf grad; the tied embedding gets input + lm_head grads."""
        for k, v in P.items():
            g = self.leaves[k].grad
            if k == "model.embed_tokens.weight":
                g = g + self.leaves["lm_head.weight"].grad
            v.grad = g.float().clone()
