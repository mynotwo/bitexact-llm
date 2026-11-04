import os, sys, torch
exec(open(os.path.join(os.path.dirname(__file__), "sweep.py")).read().split("cases = ")[0])
for R, ctx in [(1,512),(1,1024),(1,2048),(4,1024),(8,1024),(8,2048),(1,4096)]:
    q, qpos, c, bt = mk(R, ctx)
    print(R, ctx, {ns: round(bench(lambda: T.attend(q, qpos, c, bt, tab, nsplit=ns))) for ns in (1,2,4,8,16,33)}, flush=True)
