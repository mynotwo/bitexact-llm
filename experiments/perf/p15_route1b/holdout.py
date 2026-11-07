"""p15 holdout: at held-out M values, the routed FusedLin.__call__ (current code) vs the fused _lin_kernel
(TABLE cfg, what __call__ did before routing). Checks torch.equal on outputs and flags, then graph-times the two
paths in alternating pairs. Ratio = fused / routed (>= 0.97 means the routed choice is not slower).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--const', required=True)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.tree) / 'experiments/perf/p7_exactplugin'))
    from gtime import graph_time
    from bitexact.exact.model import Lin
    from bitexact.vllm_plugin.exact_gemm import FusedLin

    constants = torch.load(args.const, map_location='cpu', weights_only=False)
    gen = torch.Generator(device='cuda').manual_seed(20261107)
    results = []
    for name in ('wqkv', 'wo', 'wgu', 'wd', 'lm'):
        lin = Lin(constants, 'lm' if name == 'lm' else f'0.{name}', 'cuda', 'triton', False)
        n, k = lin.N, lin.limbs.shape[2]
        routed = FusedLin(lin)
        for m in [int(x) for x in args.rows.split(',')]:
            fused_cfg = next(c for mx, c in FusedLin.TABLE[(n, k)] if m <= mx)
            fused = FusedLin(lin, cfg=fused_cfg)
            ok = True
            for lim, shr in ((1 << 22, (-30, -20)), (1 << 12, (-45, -8)), (3, (-70, 10))):
                q = torch.randint(-lim, lim + 1, (m, k), device='cuda', generator=gen).to(torch.int32)
                sh = torch.randint(shr[0], shr[1], (m,), device='cuda', generator=gen)
                fa = torch.zeros(1, dtype=torch.int32, device='cuda')
                fb = torch.zeros(1, dtype=torch.int32, device='cuda')
                ok &= torch.equal(fused(q, sh, fa), routed(q, sh, fb)) and torch.equal(fa, fb)
            q = torch.randint(-(1 << 22), (1 << 22) + 1, (m, k), device='cuda', generator=gen).to(torch.int32)
            sh = torch.full((m,), -26, device='cuda', dtype=torch.int64)
            fa = torch.zeros(1, dtype=torch.int32, device='cuda')
            fb = torch.zeros(1, dtype=torch.int32, device='cuda')
            calls = {'f': lambda: fused(q, sh, fa), 'r': lambda: routed(q, sh, fb)}
            ratios, ft, rt = [], [], []
            for pair in range(args.pairs):
                order = ['f', 'r'] if pair % 2 == 0 else ['r', 'f']
                got = {key: graph_time(calls[key], args.reps) for key in order}
                ft.append(got['f'])
                rt.append(got['r'])
                ratios.append(got['f'] / got['r'])
            results.append({'name': name, 'N': n, 'K': k, 'M': m, 'fused_cfg': list(fused_cfg), 'equal': bool(ok),
                            'fused_us': ft, 'routed_us': rt, 'ratios': ratios,
                            'min_ratio': min(ratios), 'median_ratio': statistics.median(ratios)})
            print(json.dumps({key: results[-1][key] for key in ('name', 'M', 'equal', 'min_ratio',
                                                                'median_ratio')}), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
