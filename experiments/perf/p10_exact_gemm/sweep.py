"""Measure exact FusedLin configurations without changing arithmetic or the production table.

Rotate real layers so the small projections do not fit entirely in L2. Every
candidate must equal the existing kernel's output and overflow flag before timing.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

from bitexact.exact.model import Lin
from bitexact.vllm_plugin.exact_gemm import FusedLin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p7_exactplugin"))
from gtime import graph_time  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("const")
    parser.add_argument("--rows", default="1,32,128")
    parser.add_argument("--names", default="wqkv,wo,wgu,wd,lm")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    constants = torch.load(args.const, map_location="cpu", weights_only=False)
    assert "lm.limbs" in constants, "exact constants required"
    generator = torch.Generator(device="cuda").manual_seed(1010)
    records = []
    for suffix in args.names.split(","):
        names = ["lm"] if suffix == "lm" else [f"{i}.{suffix}" for i in (0, 1, 2, 3, 4)]
        layers = [Lin(constants, name, "cuda", "triton", False) for name in names]
        kernels = [FusedLin(layer) for layer in layers]
        n, k = layers[0].N, layers[0].limbs.shape[2]
        for rows in map(int, args.rows.split(",")):
            for kernel in kernels:
                kernel.cfg = None
            q = torch.randint(-(1 << 22), (1 << 22) + 1, (rows, k), device="cuda", generator=generator)
            q = q.to(torch.int32)
            shift = torch.full((rows,), -26, device="cuda", dtype=torch.int64)
            flag = torch.zeros(1, device="cuda", dtype=torch.int32)
            expected = []
            for kernel in kernels:
                flag.zero_()
                expected.append((kernel(q, shift, flag), flag.clone()))

            def run(kernels=kernels):
                for kernel in kernels:
                    kernel(q, shift, flag)

            original = next(cfg for limit, cfg in FusedLin.TABLE.get((n, k), FusedLin.DEFAULT) if rows <= limit)
            baseline = graph_time(run, 5) / len(kernels)
            bms = (16,) if rows <= 8 else (16, 32, 64)
            configs = list(itertools.product(bms, (32, 64, 128), (64, 128, 256), (4,), (2, 3, 4)))
            configs = [original] + [cfg for cfg in configs if cfg != original]
            timings, errors = [], []
            for cfg in configs:
                for kernel in kernels:
                    kernel.cfg = cfg
                try:
                    for kernel, (reference, reference_flag) in zip(kernels, expected):
                        flag.zero_()
                        actual = kernel(q, shift, flag)
                        if not torch.equal(actual, reference) or not torch.equal(flag, reference_flag):
                            raise AssertionError(f"integer mismatch: {suffix} {rows} {cfg}")
                    timings.append({"cfg": cfg, "us": graph_time(run, 5) / len(kernels)})
                except AssertionError:
                    raise
                except Exception as exc:
                    errors.append({"cfg": cfg, "error": f"{type(exc).__name__}: {exc}"[:500]})
            timings.sort(key=lambda item: item["us"])
            record = {"name": suffix, "shape": [n, k], "rows": rows, "baseline_us": baseline,
                      "baseline_cfg": original, "ranked": timings, "errors": errors, "rotating_layers": len(kernels)}
            records.append(record)
            Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({key: value for key, value in record.items() if key not in ("ranked", "errors")}
                             | {"best": timings[:3], "compile_failures": len(errors)}), flush=True)
        del layers, kernels, expected
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
