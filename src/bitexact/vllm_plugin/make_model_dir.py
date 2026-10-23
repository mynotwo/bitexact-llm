"""Make a vLLM model directory for the int8 engine.

  python -m bitexact.vllm_plugin.make_model_dir --hf HF_LLAMA_3_2_1B_DIR --const int8_const.pt --out DIR

DIR gets config.json (architecture BitExactLlamaForCausalLM, bitexact_const = absolute path of the
constants, max_position_embeddings = 32768; the model extends the constants' 2048-row integer RoPE tables at
load time with the same formula, see model._extend_rope) and symlinks to the tokenizer files and the bf16
checkpoint (only so vLLM's default loader finds a weights file; the model ignores it).
"""

import argparse
import json
import os

from bitexact.vllm_plugin import ARCH

MAX_POS = 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--const", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.hf, "config.json")))
    cfg.update(architectures=[ARCH], bitexact_const=os.path.abspath(a.const), bitexact_nl=2,
               max_position_embeddings=MAX_POS)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.hf):
        if f != "config.json" and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.realpath(os.path.join(a.hf, f)), os.path.join(a.out, f))
    print(a.out)


if __name__ == "__main__":
    main()
