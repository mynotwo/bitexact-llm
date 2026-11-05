"""Check that bench_pc.py's only difference from tests/vllm/bench.py works: PC=0 turns prefix caching off.

Stubs vllm, runs bench_pc.main() twice (PC unset, PC=0) and asserts the `enable_prefix_caching` keyword the
script passes to vLLM's LLM. No GPU, no model:

  python experiments/perf/p9_merge/check_pc_flag.py
"""

import io
import json
import os
import runpy
import sys
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
seen = []


class LLM:
    def __init__(self, **kw):
        seen.append(kw)

    def generate(self, prompts, sp, use_tqdm=False):
        time.sleep(0.002)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[1] * 4)]) for _ in prompts]


def run(env):
    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = lambda **kw: kw
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kw: kw
    vllm.inputs = inputs
    stub = {"vllm": vllm, "vllm.inputs": inputs}
    saved = {k: sys.modules.get(k) for k in stub}
    sys.modules.update(stub)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    argv = sys.argv
    sys.argv = ["bench_pc.py", "--model", "/nonexistent/model", "--batch", "2", "--tag", "t"]
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            runpy.run_path(str(HERE / "bench_pc.py"), run_name="__main__")
    finally:
        sys.argv = argv
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return json.loads(out.getvalue().split("RESULT ", 1)[1])


assert run({})["batch"] == 2
assert seen[-1]["enable_prefix_caching"] is True, "default must keep bench.py behaviour (caching on)"
assert run({"PC": "0"})["batch"] == 2
assert seen[-1]["enable_prefix_caching"] is False, "PC=0 must reach LLM"
assert run({"PC": "1"})["batch"] == 2
assert seen[-1]["enable_prefix_caching"] is True
print("OK: PC unset / 1 -> caching on, PC=0 -> off")
