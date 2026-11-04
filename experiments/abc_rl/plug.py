"""Arm C_plugin: the integer sampler and the integer rescoring run in the BitExact vLLM plugin (in-process
engine, V1 multiprocessing off), with the weights replaced in place after every optimizer step.

  PlugEngine(hf_dir, const_path, work_dir, gpu_mem)
  .load(c)                  intq.quantize constants -> plugin tensors (rl.update_weights, post_bound re-checked)
  .generate(prompts, seeds, max_new, greedy)   -> (tokens, sampler log p) per request, as intq.generate
  .score(prompts, resps)    -> teacher-forced integer log p of every response token, as intq.score
Both log p values are integers at 2^-40 (float64), from the int64 logits rows (vllm_plugin.rl.int_logp).
"""

import os

import torch

EOS = 128001


class PlugEngine:
    def __init__(self, hf_dir, const_path, work_dir, gpu_mem=0.2, max_seqs=512, max_len=1024, batched=8192):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from bitexact.vllm_plugin import make_model_dir
        mdir = os.path.join(work_dir, "plug_model")
        if not os.path.exists(os.path.join(mdir, "config.json")):
            import sys
            argv, sys.argv = sys.argv, ["x", "--hf", hf_dir, "--const", const_path, "--out", mdir]
            make_model_dir.main()
            sys.argv = argv
        from vllm import LLM
        self.llm = LLM(model=mdir, max_model_len=max_len, max_num_seqs=max_seqs, gpu_memory_utilization=gpu_mem,
                       max_num_batched_tokens=batched, enable_prefix_caching=True, seed=0)
        self._rpc(lambda model: __import__("bitexact.vllm_plugin.rl", fromlist=["install"]).install())

    def _rpc(self, fn):
        def run(worker):
            mr = worker.model_runner
            return fn(mr.get_model() if hasattr(mr, "get_model") else mr.model)
        return self.llm.collective_rpc(run)[0]

    def load(self, c):
        def f(model):
            from bitexact.vllm_plugin.rl import update_weights
            try:
                return update_weights(model.m, c)
            finally:
                model.flag.zero_()
        r = self._rpc(f)
        self.llm.reset_prefix_cache()
        return r

    def _sampling(self, **kw):
        from vllm import SamplingParams
        return SamplingParams(detokenize=False, **kw)

    def generate(self, prompts, seeds, max_new, greedy=False):
        from vllm.inputs import TokensPrompt
        from bitexact.vllm_plugin.rl import RL
        sp = [self._sampling(temperature=0.0 if greedy else 1.0, top_p=1.0, seed=s, max_tokens=max_new)
              for s in seeds]
        RL.lp = [] if not greedy else None
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        toks = [list(o.outputs[0].token_ids) for o in outs]
        if greedy:
            return toks, None
        lp = RL.take_lp()
        return toks, [[lp[(s, t)] for t in range(len(tk))] for s, tk in zip(seeds, toks)]

    def score(self, prompts, resps):
        from vllm.inputs import TokensPrompt
        from bitexact.vllm_plugin.rl import RL
        n = len(prompts)
        sp = [self._sampling(temperature=1.0, seed=1, max_tokens=1, skip_reading_prefix_cache=True,
                             extra_args={"bitexact_score_from": len(p)}) for p in prompts]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in zip(prompts, resps)], sp,
                                 use_tqdm=False)
        res = [RL.take_scores(o.request_id).cpu().tolist() for o in outs]
        assert all(len(x) == len(r) for x, r in zip(res, resps))
        return res
