"""End to end: an overflow flagged on the LAST compute_logits of a generate must not return tokens.
python e2e_overflow.py --model INT_DIR --data GSM_DIR [--at N]   (N = step to poison, default = last)
Prints PASS when generate raises and returned nothing; with --at 0 (no poison) prints the clean run's token count."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--at", type=int, default=-1, help="compute_logits call to poison (1-based; -1 = last, 0 = none)")
    a = ap.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, 4)
    from bitexact.vllm_plugin import register
    from bitexact.vllm_plugin.model import BitExactLlamaForCausalLM
    from bitexact.vllm_plugin.sampler import IntSampler
    register()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.gen + 600, max_num_seqs=2, gpu_memory_utilization=0.3,
              enable_prefix_caching=False, seed=0, logits_processors=[IntSampler])
    at = a.gen if a.at < 0 else a.at
    orig, n = BitExactLlamaForCausalLM.compute_logits, [0]

    def patched(self, h):
        n[0] += 1
        if n[0] == at:
            self.m.err.fill_(True)               # as if the step's integer bound had been violated
        return orig(self, h)

    BitExactLlamaForCausalLM.compute_logits = patched
    ps = [TokensPrompt(prompt_token_ids=prompts[0][0])]
    sp = [SamplingParams(temperature=1.0, seed=1, max_tokens=a.gen, ignore_eos=True)]
    try:
        out = llm.generate(ps, sp, use_tqdm=False)
    except BaseException as e:                   # noqa: BLE001
        print(f"RAISED {type(e).__name__}: {str(e)[:120]!r}; compute_logits calls {n[0]}")
        print("PASS" if at > 0 else "FAIL")
        return
    print(f"RETURNED {len(out[0].outputs[0].token_ids)} tokens; compute_logits calls {n[0]}")
    print("PASS" if at == 0 else "FAIL")


if __name__ == "__main__":
    main()
