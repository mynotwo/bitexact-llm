"""Real plugin decode in vLLM (BitExact model, vLLM's FULL_DECODE_ONLY CUDA graphs).

  python prof_vllm_decode.py --model INT_DIR --data GSM_DIR --B 64 --gen 512 --prof-gen 128 --out OUT.json

As abc_measure/speed_sampler.py (bitexact engine, same prompts, T = 1, ignore_eos, prefix cache on), one LLM with
max_num_seqs = B, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0). One untimed warm-up generate, 3 timed generates of
--gen tokens, then one generate of --prof-gen tokens under torch.profiler (CUPTI also records the kernels of CUDA
graph replays). nsys could not run on 138 (read-only /tmp: "Cannot create CUDA config file").
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--prof-gen", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, 64)
    kw = {}
    from importlib.metadata import entry_points
    if not any(e.name == "bitexact_sampler" for e in entry_points(group="vllm.logits_processors")):
        from bitexact.vllm_plugin import register        # no installed package: register in-process (as portability)
        from bitexact.vllm_plugin.sampler import IntSampler
        register()
        kw["logits_processors"] = [IntSampler]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.gen + 600, max_num_seqs=a.B, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0, **kw)

    def run(rep, gen=None):
        gen = gen or a.gen
        ps = [TokensPrompt(prompt_token_ids=prompts[i % 64][0]) for i in range(a.B)]
        sp = [SamplingParams(temperature=1.0, top_p=1.0, seed=1000 * rep + i + 1, max_tokens=gen, ignore_eos=True)
              for i in range(a.B)]
        torch.cuda.synchronize()
        t0 = time.time()
        o = llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.synchronize()
        return sum(len(x.outputs[0].token_ids) for x in o), time.time() - t0

    run(0)
    reps = [run(1), run(2), run(3)]
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pn, pdt = run(4, a.prof_gen)
    trace = a.out[:-5] + ".trace.json"
    prof.export_chrome_trace(trace)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import collections
    import instrument
    at = instrument.attribute(trace)
    cls = collections.defaultdict(lambda: [0.0, 0])
    top = collections.defaultdict(lambda: [0.0, 0])
    for name, dur, path, op, dims, ann in at["kernels"]:
        cls[instrument.kernel_class(name)][0] += dur
        cls[instrument.kernel_class(name)][1] += 1
        key = f"{name[:200]} @ {ann} / {op}"
        top[key][0] += dur
        top[key][1] += 1
    if not os.environ.get("KEEP_TRACE"):
        os.remove(trace)
    row = {"B": a.B, "gen": a.gen, "tokens": reps[0][0], "sec": [round(d, 3) for _, d in reps],
           "tok_per_s_median": round(statistics.median(n / d for n, d in reps), 1),
           "ms_per_step_median": round(statistics.median(d for _, d in reps) / a.gen * 1e3, 3),
           "prompt_lens": [len(prompts[i % 64][0]) for i in range(a.B)][:8], "gpu": torch.cuda.get_device_name(0),
           "profile": {"gen": a.prof_gen, "wall_us": round(pdt * 1e6), "busy_us": round(at["busy_us"]),
                       "span_us": round(at["span_us"]), "n_kernels": at["n_kernels"],
                       "by_kernel_class": {k: [round(v[0], 1), v[1]] for k, v in sorted(cls.items(), key=lambda x: -x[1][0])},
                       "top_kernels": [[k, round(v[0], 1), v[1]] for k, v in sorted(top.items(), key=lambda x: -x[1][0])[:25]]}}
    print(json.dumps({k: v for k, v in row.items() if k != "profile"}), flush=True)
    json.dump(row, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
