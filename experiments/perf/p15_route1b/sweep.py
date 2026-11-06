"""p15: 1B exact linears, current fused _lin_kernel (TABLE cfg) vs stacked GEMM (cfg None and wgmma cfg).

For each (shape, M): integer outputs and overflow flags are compared with torch.equal on three input/shift ranges
(the ranges of tests/test_vllm_exact_gemm.py), then CUDA-graph timings are taken in alternating pairs
(order reversed on odd pairs). Ratio = fused / variant, so > 1 means the variant is faster.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

WGMMA_CFG = (128, 128, 128, 4, 8, 5)
RANGES = ((1 << 22, (-30, -20)), (1 << 12, (-45, -8)), (3, (-70, 10)))


def table_cfg(fused_cls, n, k, m):
    return next(c for mx, c in fused_cls.TABLE[(n, k)] if m <= mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True, help='git archive root with src/ and experiments/perf/p7_exactplugin')
    ap.add_argument('--const', required=True)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--names', default='wqkv,wo,wgu,wd,lm')
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.tree) / 'experiments/perf/p7_exactplugin'))
    from gtime import graph_time
    from bitexact.exact.model import Lin
    from bitexact.vllm_plugin.exact_gemm import FusedLin, _stacked_linear

    constants = torch.load(args.const, map_location='cpu', weights_only=False)
    gen = torch.Generator(device='cuda').manual_seed(20261106)
    rows = [int(x) for x in args.rows.split(',')]
    results = []
    for name in args.names.split(','):
        names = ['lm'] if name == 'lm' else [f'{i}.{name}' for i in range(args.layers)]
        lins = [Lin(constants, n, 'cuda', 'triton', False) for n in names]
        n, k = lins[0].N, lins[0].limbs.shape[2]
        for m in rows:
            fused = [FusedLin(lin, cfg=table_cfg(FusedLin, n, k, m)) for lin in lins]
            variants = {
                'stack_none': lambda f, q, sh, fl: _stacked_linear(f, q, sh, fl, True, None),
                'stack_wgmma': lambda f, q, sh, fl: _stacked_linear(f, q, sh, fl, True, WGMMA_CFG),
            }
            # Equality on all three input/shift ranges, per layer, outputs and flags.
            checks, equal, flags_equal = 0, True, True
            flags_set = 0
            for lim, shr in RANGES:
                q = torch.randint(-lim, lim + 1, (m, k), device='cuda', generator=gen).to(torch.int32)
                sh = torch.randint(shr[0], shr[1], (m,), device='cuda', generator=gen)
                for f in fused:
                    fa = torch.zeros(1, dtype=torch.int32, device='cuda')
                    ya = f(q, sh, fa)
                    flags_set += int(fa.item())
                    for vname, fn in variants.items():
                        fb = torch.zeros(1, dtype=torch.int32, device='cuda')
                        yb = fn(f, q, sh, fb)
                        equal &= torch.equal(ya, yb)
                        flags_equal &= torch.equal(fa, fb)
                        checks += 1
            for vname, fn in variants.items():
                # Timing: fused (current dispatch) vs variant, all layers in one graph, alternating pairs.
                q = torch.randint(-(1 << 22), (1 << 22) + 1, (m, k), device='cuda', generator=gen).to(torch.int32)
                sh = torch.full((m,), -26, device='cuda', dtype=torch.int64)
                fa = torch.zeros(1, dtype=torch.int32, device='cuda')
                fb = torch.zeros(1, dtype=torch.int32, device='cuda')

                def run_fused(fused=fused, q=q, sh=sh, fa=fa):
                    for f in fused:
                        f(q, sh, fa)

                def run_var(fused=fused, q=q, sh=sh, fb=fb, fn=fn):
                    for f in fused:
                        fn(f, q, sh, fb)

                fused_us, var_us, ratios = [], [], []
                for pair in range(args.pairs):
                    order = [('f', run_fused), ('v', run_var)]
                    if pair % 2:
                        order.reverse()
                    got = {}
                    for tag, fn_t in order:
                        got[tag] = graph_time(fn_t, args.reps) / len(lins)
                    fused_us.append(got['f'])
                    var_us.append(got['v'])
                    ratios.append(got['f'] / got['v'])
                rec = {'name': name, 'N': n, 'K': k, 'M': m, 'layers': len(lins), 'variant': vname,
                       'fused_cfg': list(table_cfg(FusedLin, n, k, m)),
                       'equal_checks': checks, 'equal': bool(equal), 'flags_equal': bool(flags_equal),
                       'flags_set_fused': flags_set,
                       'fused_us': fused_us, 'var_us': var_us, 'ratios': ratios,
                       'min_ratio': min(ratios), 'median_ratio': statistics.median(ratios)}
                results.append(rec)
                Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps({k: rec[k] for k in ('name', 'M', 'variant', 'equal', 'flags_equal',
                                                       'min_ratio', 'median_ratio')}), flush=True)
        del fused, lins
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
