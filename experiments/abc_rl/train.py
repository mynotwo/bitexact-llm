"""Minimal GRPO for the A/B/C train/infer arithmetic experiment (Llama-3.2-1B, GSM8K train).

  python train.py --arm {A,A_TIS,B,B_TIS,B_fp32,C,D} --seed S --steps 200 --out RUN_DIR [--hf HF_DIR --const INT8_CONST --data GSM_DIR]

Every step: fp32 master -> BitExact int8 constants (intq.quantize) -> the same quantized weights go to the
integer engine and (dequantized, bf16) to vLLM and the HF trainer. Arms differ only in where the sampler
log p and the trainer log p values come from:
  A      sample vLLM bf16, ratio = pi_float_train / pi_vllm, PPO clip
  A_TIS  sample vLLM bf16, ratio = pi_train / sg(pi_train) (= 1) times w = min(sg(pi_train / pi_vllm), 2)
  B      sample integer engine, ratio = pi_float_train / pi_int_sampler, PPO clip
  B_TIS  B with the A_TIS correction;  B_fp32  B with the trainer forward in fp32 (TF32 off)
  D      vLLM bf16 with VLLM_BATCH_INVARIANT=1 and TRITON_ATTN; trainer log p values = vLLM's own teacher-forced forward
         (prompt_logprobs), gradient of HF bf16: logp_hf + sg(logp_vllm - logp_hf)
  C      sample integer engine, trainer log p = logp_float + sg(logp_int_rescored - logp_float), PPO clip
Integer implementation (--int-impl, arms B*, C): old = intq.generate / intq.score (padded batches, one process);
new = the vLLM plugin (CUDA graphs, tri_paged) samples and bitexact.int8.scorer.Scorer (packed rows) rescores,
with the same integer log p definition (intq.int_logp).
One optimizer step per RL step (512 rollouts, gradient accumulation over microbatches), token-mean loss,
group-normalized advantages, no reference KL (beta = 0; the logged KL is sampler vs trainer).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
import floatm  # noqa: E402
import intq  # noqa: E402


def train_prompts(data, tok):
    head = "".join(f"Question: {ex['question']}\nAnswer: {common._clean(ex['answer'])}\n\n"
                   for ex in [common._jsonl(f"{data}/train.jsonl")[i] for i in common.SHOTS])
    out = []
    for i, ex in enumerate(common._jsonl(f"{data}/train.jsonl")):
        if i in common.SHOTS:
            continue
        ids = tok(head + f"Question: {ex['question']}\nAnswer:", add_special_tokens=True)["input_ids"]
        out.append((ids, ex["answer"].split("####")[-1].strip()))
    return out


def batches(n, per, seed):
    rng, order = random.Random(seed), []
    while True:
        if not order:
            order = list(range(n))
            rng.shuffle(order)
        take, order = order[:per], order[per:]
        yield take


def stats(x):
    a = x.abs()
    return {"mean_abs": float(a.mean()), "max_abs": float(a.max()), "frac_nonzero": float((x != 0).float().mean()),
            "p99_abs": float(a.float().quantile(0.99)) if a.numel() < 16_000_000 else None}


def sha_f64(xs):
    import numpy as np
    return hashlib.sha256(np.asarray(xs, dtype=np.float64).tobytes()).hexdigest()[:16]


class PluginSampler:
    """New integer sampler: the BitExact vLLM plugin in this process. Its IntLlama (MODEL.m) is a second copy of the
    int8 weights, updated in place every step (CUDA graphs keep their pointers). The sampler log p of every sampled
    token is intq.int_logp of the integer logits row, computed on the device inside the plugin's logits processor."""

    def __init__(self, hf, const, work, gpu_mem, max_seqs, max_len=1024):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        mdir = os.path.join(work, "int_model")
        os.makedirs(mdir, exist_ok=True)
        cfg = json.load(open(os.path.join(hf, "config.json")))
        cfg.update(architectures=["BitExactLlamaForCausalLM"], bitexact_const=os.path.abspath(const), bitexact_nl=2,
                   max_position_embeddings=32768)
        json.dump(cfg, open(os.path.join(mdir, "config.json"), "w"), indent=2)
        for f in os.listdir(hf):
            if f != "config.json" and not os.path.lexists(os.path.join(mdir, f)):
                os.symlink(os.path.realpath(os.path.join(hf, f)), os.path.join(mdir, f))
        from vllm import LLM
        from bitexact.vllm_plugin import register, sampler
        register()
        self.rows = []                                           # (seeds, ts, device log p) per sampler call

        def _log(s, li, seeds, ts, y):
            from bitexact.vllm_plugin.model import MODEL
            self.rows.append((seeds, ts, intq.int_logp(li, y, MODEL.m.ln)))
        sampler.IntSampler._log = _log
        orig_init = sampler.IntSampler.__init__

        def _init(s, *a, **k):
            orig_init(s, *a, **k)
            s.log = "device"                                     # truthy: apply() calls _log
        sampler.IntSampler.__init__ = _init
        self.llm = LLM(model=mdir, dtype="bfloat16", max_model_len=max_len, max_num_seqs=max_seqs,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0,
                       logits_processors=[sampler.IntSampler])
        from bitexact.vllm_plugin.model import MODEL
        self.eng = intq.IntEngine.__new__(intq.IntEngine)
        self.eng.m, self.eng.prev = MODEL.m, None

    def load(self, c):
        self.eng.load(c)
        self.llm.reset_prefix_cache()                            # KV of the previous weights

    def generate(self, prompts, seeds, max_new):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        self.rows.clear()
        sp = [SamplingParams(temperature=1.0, top_p=1.0, seed=s, max_tokens=max_new) for s in seeds]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        lp = {}
        for seeds_, ts, d in self.rows:
            v = d.tolist()
            for r, (sd, t) in enumerate(zip(seeds_, ts)):
                if t >= 0 and sd > 0:
                    lp[(sd, t)] = v[r]
        toks = [list(o.outputs[0].token_ids) for o in outs]
        lps = [[lp[(s, t)] for t in range(len(r))] for s, r in zip(seeds, toks)]
        self.eng.check()
        return toks, lps


def packed_score(eng, scorer, prompts, resps, chunk):
    """Teacher-forced integer log p with the packed Scorer, in groups of `chunk` sequences (sorted by length)."""
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) + len(resps[i]))
    for b0 in range(0, len(order), chunk):
        idx = order[b0:b0 + chunk]
        for i, (lp, _) in zip(idx, scorer.score([(prompts[i], resps[i]) for i in idx])):
            out[i] = lp.tolist()
    return out


def evaluate(eng, vs, tok, test, max_new, bs):
    prompts = [p for p, _ in test]
    res = {}
    t0 = time.time()
    it, _ = intq.generate(eng, prompts, [0] * len(prompts), max_new, greedy=True, bs=bs)
    texts = tok.batch_decode([[x for x in t if x != intq.EOS] for t in it])
    res["int_acc"] = sum(common.grade(t, g)[0] for t, (_, g) in zip(texts, test)) / len(test)
    res["int_sec"] = time.time() - t0
    if vs is None:
        return res
    t0 = time.time()
    _, _, ftexts = vs.generate(prompts, [0] * len(prompts), max_new, greedy=True)
    res["float_acc"] = sum(common.grade(t, g)[0] for t, (_, g) in zip(ftexts, test)) / len(test)
    res["float_sec"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "A_TIS", "B", "B_TIS", "B_fp32", "C", "D"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--tis-c", type=float, default=2.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--mb", type=int, default=16, help="trainer microbatch (sequences)")
    ap.add_argument("--gen-bs", type=int, default=256)
    ap.add_argument("--score-bs", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=25, help="0 = never evaluate")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true", help="deterministic torch kernels (needs CUBLAS_WORKSPACE_CONFIG)")
    ap.add_argument("--int-impl", choices=["old", "new"], default="old", help="integer sampler / scorer (arms B*, C)")
    ap.add_argument("--plugin-seqs", type=int, default=512, help="vLLM max_num_seqs of the plugin (--int-impl new)")
    ap.add_argument("--plugin-gpu-mem", type=float, default=0.20)
    ap.add_argument("--score-chunk", type=int, default=64, help="sequences per Scorer.score call (--int-impl new)")
    ap.add_argument("--gpu-mem", type=float, default=0.18)
    ap.add_argument("--d-attn", default="TRITON_ATTN", help="vLLM attention backend for arm D")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--const", required=True, help="original int8 constants (tables only)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True, help="shared dir for the dequantized vLLM model dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    train = train_prompts(a.data, tok)
    test = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, a.eval_n)

    sd = load_file(os.path.join(a.hf, "model.safetensors"))
    P = {k: v.float().to(dev).requires_grad_(True) for k, v in sd.items() if k != "lm_head.weight"}
    del sd
    c, fw, qst = intq.quantize(P)
    mdir = floatm.write_model_dir(a.hf, fw, os.path.join(a.work, "deq_model"))
    new_int = a.int_impl == "new" and a.arm in ("B", "B_TIS", "B_fp32", "C")
    vs = pl = scorer = None
    if new_int:
        pl = PluginSampler(a.hf, a.const, a.work, a.plugin_gpu_mem, a.plugin_seqs)        # before the trainer allocates
    else:
        vs = floatm.VllmSampler(mdir, a.gpu_mem, max_seqs=512, batch_invariant=a.arm == "D",
                                attn_backend=a.d_attn if a.arm == "D" else None)
    eng = intq.IntEngine(a.const, c, dev)
    if new_int:                     # Scorer re-specialises m.sg, so it gets its own engine, not the plugin's
        from bitexact.int8 import scorer as scorer_mod
        scorer = scorer_mod.Scorer(eng.m)
        scorer_mod.logp_rows = lambda lg, tgt: intq.int_logp(lg, tgt, eng.m.ln)      # integer log p, as the sampler
    tr = floatm.FloatTrainer(mdir, dev, torch.float32 if a.arm == "B_fp32" else torch.bfloat16)
    torch.backends.cudnn.allow_tf32 = False
    opt = torch.optim.AdamW(list(P.values()), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    log = open(os.path.join(a.out, "steps.jsonl"), "a")
    evlog = open(os.path.join(a.out, "eval.jsonl"), "a")
    gen = batches(len(train), a.prompts, a.seed)
    int_sample = a.arm in ("B", "B_TIS", "B_fp32", "C")
    print(f"start arm {a.arm} seed {a.seed} mem {torch.cuda.memory_allocated() / 2**30:.1f}G", flush=True)

    for step in range(a.steps + 1):
        t = {}
        t0 = time.time()
        if step > 0:
            c, fw, qst = intq.quantize(P)
        changed = eng.load(c)
        if vs is not None:
            vs.load(fw)
        if pl is not None:
            pl.load(c)
        tr.set_weights(fw)
        torch.cuda.synchronize()
        t["quantize"] = time.time() - t0
        bound_err = None
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            try:
                ev = evaluate(eng, vs, tok, test, a.max_new, a.gen_bs)
            except OverflowError as e:
                ev, bound_err = {"error": str(e)}, str(e)
            ev["step"] = step
            evlog.write(json.dumps(ev) + "\n")
            evlog.flush()
            print("EVAL", json.dumps(ev), flush=True)
        if step == a.steps:
            break

        # ---- sample ----
        t0 = time.time()
        pidx = next(gen)
        prompts = [train[i][0] for i in pidx for _ in range(a.G)]
        golds = [train[i][1] for i in pidx for _ in range(a.G)]
        seeds = [1 + (a.seed * 1000 + step) * 10000 + j for j in range(len(prompts))]
        try:
            if int_sample:
                if pl is not None:
                    resps, slp = pl.generate(prompts, seeds, a.max_new)
                else:
                    resps, slp = intq.generate(eng, prompts, seeds, a.max_new, bs=a.gen_bs)
                texts = tok.batch_decode([[x for x in r if x != intq.EOS] for r in resps])
            else:
                resps, slp, texts = vs.generate(prompts, seeds, a.max_new)
        except OverflowError as e:
            bound_err = str(e)
            print("BOUND", step, e, flush=True)
            log.write(json.dumps({"step": step, "bound_error": bound_err}) + "\n")
            continue
        keep = [i for i, r in enumerate(resps) if len(r) > 0]
        rew = torch.tensor([float(common.grade(x, g)[0]) for x, g in zip(texts, golds)])
        rg = rew.view(-1, a.G)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, unbiased=False, keepdim=True) + 1e-4)).view(-1)
        torch.cuda.synchronize()
        t["sample"] = time.time() - t0

        # ---- integer rescoring (C) ----
        t0 = time.time()
        ilp = None
        if a.arm == "C":
            kp, kr = [prompts[i] for i in keep], [resps[i] for i in keep]
            ilp = packed_score(eng, scorer, kp, kr, a.score_chunk) if scorer is not None else \
                intq.score(eng, kp, kr, a.score_bs)
        elif a.arm == "D":
            ilp = vs.score([prompts[i] for i in keep], [resps[i] for i in keep])
        if scorer is not None:
            scorer.c = []                                 # paged K / V cache of the scorer: rebuilt on the next call
            torch.cuda.empty_cache()                      # return the scorer's temporaries before the trainer runs
        t["score_int"] = time.time() - t0

        # ---- trainer forward / backward ----
        t0 = time.time()
        ntok = sum(len(resps[i]) for i in keep)
        acc = {k: [] for k in ("delta", "ratio", "clipped", "w", "fi")}
        loss_sum = torch.zeros((), dtype=torch.float64, device=dev)
        order = sorted(keep, key=lambda i: len(prompts[i]) + len(resps[i]))
        pos_in_keep = {i: j for j, i in enumerate(keep)}
        for b0 in range(0, len(order), a.mb):
            mb = order[b0:b0 + a.mb]
            lpf, spans = tr.logp([prompts[i] for i in mb], [resps[i] for i in mb], grad=True)
            sl = torch.tensor([x for i in mb for x in slp[i]], device=dev, dtype=torch.float64)
            A_ = torch.tensor([float(adv[i]) for i in mb for _ in resps[i]], device=dev)
            lpf64 = lpf.detach().double()
            if a.arm in ("C", "D"):
                il = torch.tensor([x for i in mb for x in ilp[pos_in_keep[i]]], device=dev, dtype=torch.float64)
                acc["fi"].append(lpf64 - il)                  # float vs int trainer gap (diagnostic)
                lp_tr = lpf + (il - lpf64).float().detach()   # value = int log p, gradient of float
                delta = il - sl
            else:
                lp_tr = lpf
                delta = lpf64 - sl
            if a.arm in ("A_TIS", "B_TIS"):
                w = torch.exp(delta).clamp(max=a.tis_c).float()
                ratio = torch.exp(lpf - lpf.detach())
                acc["w"].append(w)
            else:
                ratio = torch.exp(lp_tr - sl.float())
                w = torch.ones_like(ratio)
            s1, s2 = ratio * A_, ratio.clamp(1 - a.eps, 1 + a.eps) * A_
            loss = -(w * torch.minimum(s1, s2)).sum() / ntok
            loss.backward()
            loss_sum += loss.detach().double()
            acc["delta"].append(delta)
            acc["ratio"].append(ratio.detach())
            acc["clipped"].append((s2 < s1).detach() & (A_ != 0))
        t["backward"] = time.time() - t0

        t0 = time.time()
        tr.grads_to_master(P)
        gnorm = float(torch.nn.utils.clip_grad_norm_(list(P.values()), a.clip_grad))
        finite = math.isfinite(gnorm)
        if finite:
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t["optim"] = time.time() - t0

        d = torch.cat(acc["delta"])
        r = torch.exp(d)
        rec = {"step": step, "reward": float(rew.mean()), "n_tok": ntok,
               "resp_len": ntok / max(1, len(keep)), "groups_with_signal": int((rg.std(1) > 0).sum()),
               "delta": stats(d), "ratio_outside_clip": float(((r < 1 - a.eps) | (r > 1 + a.eps)).float().mean()),
               "clip_frac": float(torch.cat(acc["clipped"]).float().mean()),
               "kl_k3": float((r - 1 - d).mean()), "kl_k1": float((-d).mean()),
               "grad_norm": gnorm, "grad_finite": finite, "int8_changed_frac": changed,
               "fm_min": min(qst["fm"]), "fm_max": max(qst["fm"]), "time": t,
               "step_sec": sum(t.values()), "mem_gb": torch.cuda.max_memory_allocated() / 2**30,
               "loss": float(loss_sum), "loss_hex": float(loss_sum).hex(),
               "tok_sha": hashlib.sha256(json.dumps([resps[i] for i in keep]).encode()).hexdigest()[:16],
               "slp_sha": sha_f64([x for i in keep for x in slp[i]]),
               "ilp_sha": sha_f64([x for l in ilp for x in l]) if ilp is not None else None}
        if acc["w"]:
            w = torch.cat(acc["w"])
            rec["tis_w_mean"], rec["tis_truncated_frac"] = float(w.mean()), float((w >= a.tis_c).float().mean())
        if acc["fi"]:
            rec["float_vs_int_trainer"] = stats(torch.cat(acc["fi"]))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step} R {rec['reward']:.4f} |d| {rec['delta']['mean_abs']:.2e} clip {rec['clip_frac']:.4f} "
              f"gn {gnorm:.3f} chg {changed:.2e} t " + " ".join(f"{k}={v:.1f}" for k, v in t.items()), flush=True)
        if step == 0:
            with open(os.path.join(a.out, "samples_step0.json"), "w") as f:
                json.dump([{"text": texts[i], "gold": golds[i], "reward": float(rew[i])} for i in range(8)], f)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
