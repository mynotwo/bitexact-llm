"""auto-rule attend() vs base tri_paged (BASE_SRC) on random caches: equality + time."""
import os, sys, torch, importlib.util
sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(__file__), "sweep.py")).read().split("cases = ")[0])
spec = importlib.util.spec_from_file_location("tri_base", os.environ["BASE_SRC"] + "/bitexact/vllm_plugin/tri_paged.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
for R, ctx in [(1,192),(1,1024),(1,4096),(1,16384),(1,32768),(8,192),(8,4096),(8,16384),(8,32768),(16,16384),(32,192),(32,4096),(40,192),(64,192),(64,2048),(128,192),(128,1024)]:
    q, qpos, c, bt = mk(R, ctx)
    a, b = T.attend(q, qpos, c, bt, tab), B.attend(q, qpos, c, bt, tab)
    eq = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    print(f"R={R} ctx={ctx} base {bench(lambda: B.attend(q, qpos, c, bt, tab)):.0f}us new {bench(lambda: T.attend(q, qpos, c, bt, tab)):.0f}us equal {eq}", flush=True)
    del c
