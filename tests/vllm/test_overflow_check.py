"""The bound-violation flag reaches the user even when only the engine's last step sets it (needs CUDA and a
plugin model directory). Run: BITEXACT_VLLM_MODEL=DIR pytest -q tests/vllm/test_overflow_check.py

One request, max_tokens=N: N compute_logits calls (prefill + N-1 decodes). A clean run returns N tokens; then
the flag is forced on the device in the N-th (last) call only, and generate must raise instead of returning.
"""

import os

import pytest
import torch

MODEL_DIR = os.environ.get("BITEXACT_VLLM_MODEL")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not (torch.cuda.is_available() and MODEL_DIR),
                                                  reason="needs CUDA + BITEXACT_VLLM_MODEL")]
N = 4


def test_overflow_in_last_step_raises():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"         # the model lives in this process
    os.environ.pop("BITEXACT_HASH_LOG", None)                    # the hot-path (non-blocking) check
    from vllm import LLM, SamplingParams
    from bitexact.vllm_plugin import model as pm
    llm = LLM(model=MODEL_DIR, max_model_len=2048, gpu_memory_utilization=0.35, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=N, ignore_eos=True)
    out = llm.generate([[1, 2, 3, 4, 5]], sp)
    assert len(out[0].outputs[0].token_ids) == N

    calls = []
    orig = pm.BitExactLlamaForCausalLM.compute_logits

    def forced(self, hidden_states):
        calls.append(1)
        if len(calls) == N:
            self.m.err.fill_(True)                              # device-side, like a real violation
        return orig(self, hidden_states)
    pm.BitExactLlamaForCausalLM.compute_logits = forced
    try:
        with pytest.raises(Exception) as e:
            llm.generate([[6, 7, 8, 9, 10]], sp)
    finally:
        pm.BitExactLlamaForCausalLM.compute_logits = orig
    assert len(calls) == N, f"flag must be set in the last step only ({len(calls)} calls)"
    chain, x = [], e.value
    while x is not None:
        chain.append(f"{type(x).__name__}: {x}")
        x = x.__cause__ or x.__context__
    assert any("integer bound violated" in c for c in chain), chain
