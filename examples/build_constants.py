"""Build an engine's integer constants from a HF Llama-3.2-1B checkpoint (CPU, float64, done once).

  python examples/build_constants.py --model PATH/Llama-3.2-1B --mode exact --out exact_const.pt
  python examples/build_constants.py --model PATH/Llama-3.2-1B --mode int8 --out int8_const.pt

The printed sha256 identifies the integer model; files built from the same checkpoint on different
machines or torch versions should print the same value. The exact file is ~4.8 GB.
"""

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF directory with config.json and model.safetensors")
    ap.add_argument("--mode", choices=["exact", "int8"], default="exact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="first N layers only (tests)")
    a = ap.parse_args()
    if a.mode == "exact":
        from bitexact.exact.consts import build, summary
        c = build(a.model, a.layers)
        info = summary(c)
    else:
        from bitexact.int8.consts import build
        c = build(a.model, a.layers)
        info = {"layers": c["n_layers"], "sha256": c["sha256"]}
    torch.save(c, a.out)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
