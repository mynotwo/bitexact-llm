"""Check selected exact GEMM configurations at bucket boundaries and measure paired timings."""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

from bitexact.exact.model import Lin
from bitexact.vllm_plugin.exact_gemm import FusedLin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'p7_exactplugin'))
from gtime import graph_time  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('const')
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rows', default='1,8,9,16,17,31,32,33,64,65,96,127,128,129,256,512')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('baseline_exact_gemm', args.base)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    constants = torch.load(args.const, map_location='cpu', weights_only=False)
    assert 'lm.limbs' in constants
    generator = torch.Generator(device='cuda').manual_seed(20261105)
    results = []
    for suffix in ('wqkv', 'wo', 'wgu', 'wd', 'lm'):
        names = ['lm'] if suffix == 'lm' else [f'{i}.{suffix}' for i in (0, 1, 2, 3, 4)]
        layers = [Lin(constants, name, 'cuda', 'triton', False) for name in names]
        old = [baseline.FusedLin(layer) for layer in layers]
        new = [FusedLin(layer) for layer in layers]
        n, k = layers[0].N, layers[0].limbs.shape[2]
        for rows in map(int, args.rows.split(',')):
            q = torch.randint(-(1 << 22), (1 << 22) + 1, (rows, k), device='cuda', generator=generator)
            q = q.to(torch.int32)
            shift = torch.full((rows,), -26, device='cuda', dtype=torch.int64)
            flag_a = torch.zeros(1, device='cuda', dtype=torch.int32)
            flag_b = torch.zeros_like(flag_a)
            for a, b in zip(old, new):
                flag_a.zero_()
                flag_b.zero_()
                assert torch.equal(a(q, shift, flag_a), b(q, shift, flag_b)), (suffix, rows)
                assert torch.equal(flag_a, flag_b), (suffix, rows, 'flags')

            def run_old(old=old):
                for kernel in old:
                    kernel(q, shift, flag_a)

            def run_new(new=new):
                for kernel in new:
                    kernel(q, shift, flag_b)

            a_times, b_times = [], []
            for rep in range(3):
                order = ((run_old, a_times), (run_new, b_times))
                for call, values in (order if rep % 2 == 0 else reversed(order)):
                    values.append(graph_time(call, 4) / len(layers))
            old_cfg = next(cfg for limit, cfg in baseline.FusedLin.TABLE.get((n, k), baseline.FusedLin.DEFAULT)
                           if rows <= limit)
            new_cfg = next(cfg for limit, cfg in FusedLin.TABLE.get((n, k), FusedLin.DEFAULT) if rows <= limit)
            result = {'name': suffix, 'shape': [n, k], 'rows': rows, 'old_cfg': old_cfg, 'new_cfg': new_cfg,
                      'base_us': a_times, 'repo_us': b_times,
                      'ratio': statistics.median(a_times) / statistics.median(b_times),
                      'equal_layers': len(layers)}
            results.append(result)
            Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        del old, new, layers
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
