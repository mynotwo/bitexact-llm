"""Arm D step-0 checks: is vLLM batch-invariant sampling bitwise equal to vLLM's own teacher-forced scoring?

  python check_d.py --hf HF_DIR --const INT8_CONST --data GSM_DIR --work WORK --out OUT.json [--ref OUT_OTHER.json]

Step-0 weights (dequantized int8, as every float arm), 64 GSM8K-train prompts x 8, T = 1, 256 tokens.
1. sampler log p (decode, raw_logprobs) vs scorer log p (prompt_logprobs of prompt + response): |diff|.
2. scorer log p with different batch compositions: all 512 in one call, chunks of 37, one at a time (first 32).
3. integer engine (arm C) on the same tokens: score in batches of 16 and 3.
With --ref (the JSON written on another GPU type), compares the vLLM scores and integer scores of the same
tokens across hosts. Writes tokens, sampler and scorer log p to OUT.json.
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import floatm  # noqa: E402
import intq  # noqa: E402
import train as T  # noqa: E402


def cmp(a, b):
    d = [abs(x - y) for u, v in zip(a, b) for x, y in zip(u, v)]
    return {"tokens": len(d), "nonzero": sum(x != 0 for x in d), "max": max(d), "mean": sum(d) / len(d)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("hf", "const", "data", "work", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ref")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--attn", default="TRITON_ATTN", help="vLLM attention backend (FLASH_ATTN: vLLM default)")
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.hf)
    train = T.train_prompts(a.data, tok)
    sd = load_file(os.path.join(a.hf, "model.safetensors"))
    P = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
    c, fw, _ = intq.quantize(P)
    del P
    mdir = floatm.write_model_dir(a.hf, fw, os.path.join(a.work, "deq_model"))
    vs = floatm.VllmSampler(mdir, a.gpu_mem, max_seqs=512, batch_invariant=True, attn_backend=a.attn)
    vs.load(fw)
    res = {"gpu": torch.cuda.get_device_name(0), "attn": a.attn}
    if a.ref:
        ref = json.load(open(a.ref))
        prompts, resps, slp = ref["prompts"], ref["resps"], ref["sampler_lp"]
        rt, rlp, _ = vs.generate(prompts, ref["seeds"], 256)
        res["resample_same_tokens"] = sum(x == y for x, y in zip(rt, resps))
        res["resample_vs_ref_sampler"] = cmp(rlp, slp) if res["resample_same_tokens"] == len(resps) else None
    else:
        idx = next(T.batches(len(train), 64, 1))
        prompts = [train[i][0] for i in idx for _ in range(8)]
        seeds = [1 + (1000 + 0) * 10000 + j for j in range(len(prompts))]
        resps, slp, _ = vs.generate(prompts, seeds, 256)
        res.update(prompts=prompts, resps=resps, sampler_lp=slp, seeds=seeds)
    s_all = vs.score(prompts, resps)
    s_37 = vs.score(prompts, resps, chunk=37)
    s_1 = vs.score(prompts[:32], resps[:32], chunk=1)
    res["D_sampler_vs_scorer"] = cmp(slp, s_all)
    res["D_scorer_all_vs_chunk37"] = cmp(s_all, s_37)
    res["D_scorer_all_vs_single"] = cmp(s_all[:32], s_1)
    res["scorer_lp"] = s_all
    eng = intq.IntEngine(a.const, c, dev)
    eng.load(c)
    i16 = intq.score(eng, prompts, resps, bs=16)
    i3 = intq.score(eng, prompts[:48], resps[:48], bs=3)
    res["C_int_bs16_vs_bs3"] = cmp(i16[:48], i3)
    res["int_lp"] = i16
    if a.ref:
        res["D_scorer_vs_ref_host"] = cmp(s_all, ref["scorer_lp"])
        res["D_sampler_ref_host_vs_scorer_here"] = cmp(slp, s_all)
        res["C_int_vs_ref_host"] = cmp(i16, ref["int_lp"])
    for k, v in res.items():
        if isinstance(v, dict) or k in ("gpu", "attn", "resample_same_tokens"):
            print("RESULT", k, v, flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
