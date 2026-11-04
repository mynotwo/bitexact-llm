"""Run a vLLM script with the bitexact plugin registered in-process (the 138 venv has no installed bitexact
package, so no entry points): python inproc.py SCRIPT [args]. Adds IntSampler to every LLM(...) call."""
import os
import runpy
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import vllm  # noqa: E402

from bitexact.vllm_plugin import register  # noqa: E402
from bitexact.vllm_plugin.sampler import IntSampler  # noqa: E402

register()
_init = vllm.LLM.__init__


def _llm_init(self, *a, **kw):
    kw.setdefault("logits_processors", [IntSampler])
    _init(self, *a, **kw)


vllm.LLM.__init__ = _llm_init
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))      # as `python SCRIPT` would
runpy.run_path(sys.argv[0], run_name="__main__")
