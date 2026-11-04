"""Arm D CPU recomputation attempt with plain PyTorch: HF transformers on CPU, teacher forcing on prompt + response.

  python score_cpu_float.py --rollout R.json --model DEQ_DIR --data GSM_DIR --dtype bfloat16|float32
         --seeds-from FILE --out S.json [--threads 32] [--batch-invariant]

Batch 1, eager attention, float32 log-softmax of the logits at the response positions (same quantity as vLLM's
raw log p). --batch-invariant first calls vLLM's batch-invariant mode initialiser (VLLM_BATCH_INVARIANT=1) to see
whether it changes CPU ops at all.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seeds-from", required=True)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    note = None
    if a.batch_invariant:
        try:
            try:
                from vllm.model_executor.determinism import batch_invariant as bi  # vLLM >= 0.3x
            except ImportError:
                from vllm.model_executor.layers import batch_invariant as bi
            bi.init_batch_invariance()
            from vllm.platforms import current_platform
            note = (f"called {bi.__name__}.init_batch_invariance; platform cuda_alike={current_platform.is_cuda_alike()}"
                    f" xpu={current_platform.is_xpu()}; aten overrides registered only for CUDA/XPU dispatch keys")
        except Exception as e:  # recorded, not fatal
            note = f"batch-invariant init failed: {type(e).__name__}: {e}"
        print(note, flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    keep = set(int(x) for x in open(a.seeds_from).read().split())
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items()) if s in keep]
    dt = getattr(torch, a.dtype)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="eager").eval()
    out, t0 = {}, time.time()
    with torch.no_grad():
        for s, p, r in items:
            ids = torch.tensor([p + r])
            lg = m(ids).logits[0, len(p) - 1:len(p) - 1 + len(r)].float()
            ls = torch.log_softmax(lg, -1)
            out[s] = {"lp": ls[torch.arange(len(r)), torch.tensor(r)].tolist(), "am": lg.argmax(-1).tolist()}
    meta = dict(vars(a), sec=round(time.time() - t0, 1), torch=torch.__version__, note=note)
    common.save(a.out, f"cpu_hf_{a.dtype}", meta, out)
    print(f"RESULT {a.out} seqs {len(out)} sec {meta['sec']}", flush=True)


if __name__ == "__main__":
    main()
