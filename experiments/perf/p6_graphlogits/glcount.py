"""Run tests/vllm/bench.py with BITEXACT_GL_CHECK=1 and count the graph-logits assertions actually taken.

The flag only turns on the assertions; this wrapper also records how many replays checked and how many of them had the
greedy gate open, so a passing run is evidence the check ran rather than evidence it was skipped.

  python experiments/perf/p6_graphlogits/glcount.py --model DIR --batch 32 --temp 0 --tag c_b32_t0
"""

import atexit
import os
import runpy
import sys

os.environ["BITEXACT_GL_CHECK"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tests", "vllm", "bench.py")

from bitexact.vllm_plugin import model as M

CNT = {"n": 0, "gate": 0}
_orig = M.BitExactLlamaForCausalLM._gl_check


def _check(self, hidden_states, head, y):
    CNT["n"] += 1
    CNT["gate"] += int(self.gate.item())
    return _orig(self, hidden_states, head, y)


M.BitExactLlamaForCausalLM._gl_check = _check
atexit.register(lambda: print("GLCOUNT checked=%d gate_on=%d" % (CNT["n"], CNT["gate"]), flush=True))

sys.argv = [BENCH] + sys.argv[1:]
runpy.run_path(BENCH, run_name="__main__")