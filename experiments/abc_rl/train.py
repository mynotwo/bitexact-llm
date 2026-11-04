"""Minimal GRPO for the A/B/C train/infer arithmetic experiment (Llama-3.2-1B, GSM8K train).

  python train.py --arm {A,A_TIS,B,C} --seed S --steps 200 --out RUN_DIR [--hf HF_DIR --const INT8_CONST --data GSM_DIR]

Every step: fp32 master -> BitExact int8 constants (intq.quantize) -> the same quantized weights go to the
integer engine and (dequantized, bf16) to vLLM and the HF trainer. Arms differ only in where the sampler
log p and the trainer log p values come from:
  A      sample vLLM bf16, ratio = pi_float_train / pi_vllm, PPO clip
  A_TIS  sample vLLM bf16, ratio = pi_train / sg(pi_train) (= 1) times w = min(sg(pi_train / pi_vllm), 2)
  B      sample integer engine, ratio = pi_float_train / pi_int_sampler, PPO clip
  C      sample integer engine, trainer log p = logp_float + sg(logp_int_rescored - logp_float), PPO clip
One optimizer step per RL step (512 rollouts, gradient accumulation over microbatches), token-mean loss,
group-normalized advantages, no reference KL (beta = 0; the logged KL is sampler vs trainer).
"""

import argparse
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


def evaluate(eng, vs, tok, test, max_new, bs):
    prompts = [p for p, _ in test]
    res = {}
    t0 = time.time()
    it, _ = intq.generate(eng, prompts, [0] * len(prompts), max_new, greedy=True, bs=bs)
    texts = tok.batch_decode([[x for x in t if x != intq.EOS] for t in it])
    res["int_acc"] = sum(common.grade(t, g)[0] for t, (_, g) in zip(texts, test)) / len(test)
    res["int_sec"] = time.time() - t0
    t0 = time.time()
    _, _, ftexts = vs.generate(prompts, [0] * len(prompts), max_new, greedy=True)
    res["float_acc"] = sum(common.grade(t, g)[0] for t, (_, g) in zip(ftexts, test)) / len(test)
    res["float_sec"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "A_TIS", "B", "C"], required=True)
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
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--gpu-mem", type=float, default=0.18)
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
    vs = floatm.VllmSampler(mdir, a.gpu_mem, max_seqs=512)          # before the trainer allocates
    eng = intq.IntEngine(a.const, c, dev)
    tr = floatm.FloatTrainer(mdir, dev)
    opt = torch.optim.AdamW(list(P.values()), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    log = open(os.path.join(a.out, "steps.jsonl"), "a")
    evlog = open(os.path.join(a.out, "eval.jsonl"), "a")
    gen = batches(len(train), a.prompts, a.seed)
    int_sample = a.arm in ("B", "C")
    print(f"start arm {a.arm} seed {a.seed} mem {torch.cuda.memory_allocated() / 2**30:.1f}G", flush=True)

    for step in range(a.steps + 1):
        t = {}
        t0 = time.time()
        if step > 0:
            c, fw, qst = intq.quantize(P)
        changed = eng.load(c)
        vs.load(fw)
        tr.set_weights(fw)
        torch.cuda.synchronize()
        t["quantize"] = time.time() - t0
        bound_err = None
        if step % a.eval_every == 0 or step == a.steps:
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
            ilp = intq.score(eng, [prompts[i] for i in keep], [resps[i] for i in keep], a.score_bs)
        t["score_int"] = time.time() - t0

        # ---- trainer forward / backward ----
        t0 = time.time()
        ntok = sum(len(resps[i]) for i in keep)
        acc = {k: [] for k in ("delta", "ratio", "clipped", "w", "fi")}
        order = sorted(keep, key=lambda i: len(prompts[i]) + len(resps[i]))
        pos_in_keep = {i: j for j, i in enumerate(keep)}
        for b0 in range(0, len(order), a.mb):
            mb = order[b0:b0 + a.mb]
            lpf, spans = tr.logp([prompts[i] for i in mb], [resps[i] for i in mb], grad=True)
            sl = torch.tensor([x for i in mb for x in slp[i]], device=dev, dtype=torch.float64)
            A_ = torch.tensor([float(adv[i]) for i in mb for _ in resps[i]], device=dev)
            lpf64 = lpf.detach().double()
            if a.arm == "C":
                il = torch.tensor([x for i in mb for x in ilp[pos_in_keep[i]]], device=dev, dtype=torch.float64)
                acc["fi"].append(lpf64 - il)                  # float vs int trainer gap (diagnostic)
                lp_tr = lpf + (il - lpf64).float().detach()   # value = int log p, gradient of float
                delta = il - sl
            else:
                lp_tr = lpf
                delta = lpf64 - sl
            if a.arm == "A_TIS":
                w = torch.exp(delta).clamp(max=a.tis_c).float()
                ratio = torch.exp(lpf - lpf.detach())
                acc["w"].append(w)
            else:
                ratio = torch.exp(lp_tr - sl.float())
                w = torch.ones_like(ratio)
            s1, s2 = ratio * A_, ratio.clamp(1 - a.eps, 1 + a.eps) * A_
            loss = -(w * torch.minimum(s1, s2)).sum() / ntok
            loss.backward()
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
               "step_sec": sum(t.values()), "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
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
