"""Where a scorer run spends its time: torch.profiler over the first --nbatch batches (after a warm-up pass).

  python prof.py --mode packed --rollout R.json --bs 8 --nbatch 4   (same engine flags as bench.py)
Prints wall time, GPU busy time (union of kernel intervals), GPU idle share and the top kernels by total time.
"""
import collections
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402


@torch.no_grad()
def main():
    ap = bench.parser()
    ap.add_argument("--nbatch", type=int, default=4)
    ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args()
    items, m = bench.load(a)
    score = bench.make_scorer(a, m)
    bat = [[(p, r) for _, p, r, _ in items[i:i + a.bs]] for i in range(0, len(items), a.bs)][:a.nbatch]
    for b in bat:
        score(b, False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.time()
        for b in bat:
            score(b, False)
        torch.cuda.synchronize()
        wall = time.time() - t0
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    iv = sorted((e.time_range.start, e.time_range.end) for e in ev)
    busy, cur = 0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            busy += (cur[1] - cur[0]) if cur else 0
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    busy += cur[1] - cur[0]
    tot = sum(len(p) + len(r) for b in bat for p, r in b)
    print(f"tokens {tot} wall {wall * 1e3:.0f} ms ({tot / wall:.0f} tok/s) GPU busy {busy / 1e3:.0f} ms "
          f"idle {100 * (1 - busy / 1e6 / wall):.1f}% kernels {len(ev)}")
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        agg[e.name[:90]][0] += e.time_range.elapsed_us()
        agg[e.name[:90]][1] += 1
    for k, (us, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:a.top]:
        print(f"{us / 1e3:9.1f} ms {100 * us / 1e6 / wall:5.1f}% {n:6d}  {k}")


if __name__ == "__main__":
    main()
