"""Run the int8 engine inside vLLM (pip install -e ".[vllm]") and log per-position logits hashes.

  python -m bitexact.vllm_plugin.make_model_dir --hf PATH/Llama-3.2-1B --const int8_const.pt --out intmodel
  python examples/vllm_offline.py --model intmodel --out run1      # writes run1.json, run1.hash

Run it twice with different --batch-noise to check that a target's tokens and logits hashes do not depend
on what else vLLM batches with it (compare the .hash files; lines are "seed t sha16 token").
"""

import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch-noise", type=int, default=0, help="extra filler requests batched with targets")
    a = ap.parse_args()
    if os.path.exists(a.out + ".hash"):
        os.remove(a.out + ".hash")
    os.environ["BITEXACT_HASH_LOG"] = a.out + ".hash"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    rng = random.Random(17)
    targets = [[rng.randrange(1000, 120000) for _ in range(n)] for n in (7, 40, 129, 300)]
    fill = [[rng.randrange(1000, 120000) for _ in range(rng.randrange(3, 600))] for _ in range(a.batch_noise)]
    llm = LLM(model=a.model, enforce_eager=True, max_model_len=2048, enable_prefix_caching=True)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in targets + fill]
    params = [SamplingParams(temperature=(0.0, 0.7, 1.0, 0.7)[i % 4], seed=i + 1, max_tokens=a.max_new,
                             ignore_eos=True, detokenize=False) for i in range(len(targets))]
    params += [SamplingParams(temperature=0.7, seed=1000 + i, max_tokens=rng.randrange(1, 64), ignore_eos=True,
                              detokenize=False) for i in range(len(fill))]
    outs = llm.generate(prompts, params)
    with open(a.out + ".json", "w") as f:
        json.dump({str(i + 1): list(o.outputs[0].token_ids) for i, o in enumerate(outs[:len(targets)])}, f)
    print(a.out + ".json")


if __name__ == "__main__":
    main()
