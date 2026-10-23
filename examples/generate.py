"""Generate text with an integer engine and print the per-position logits hashes.

  python examples/generate.py --const exact_const.pt --tokenizer PATH/Llama-3.2-1B \
      --prompt "The lighthouse keeper wrote" --max-new 64 --temperature 0.7 --seed 1 [--device cpu]

Temperature T > 0 samples with Gumbel-max and integer noise keyed (seed, t); T = 0 is greedy.
"""

import argparse

import torch

from bitexact import generate, load
from bitexact.sampling import temp_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gemm", default="triton", help="exact mode on CUDA: triton or cublas")
    ap.add_argument("--compiled", action="store_true", help="torch.compile the element-wise stages")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    kw = {"compiled": a.compiled, **({"gemm": a.gemm} if "lm.limbs" in c else {})}
    m = load(c, a.device, **kw)
    tn, td = temp_ratio(a.temperature)
    ids = tok(a.prompt)["input_ids"]
    toks, hashes = generate(m, ids, a.max_new, tn, td, a.seed)
    print(a.prompt + tok.decode(toks))
    print("model sha256", c["sha256"][:16])
    print("last logits sha256", hashes[-1][:16])


if __name__ == "__main__":
    main()
