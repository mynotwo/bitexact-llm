"""Model-level tests need constants: BITEXACT_EXACT_CONST / BITEXACT_INT8_CONST (files from
examples/build_constants.py), or BITEXACT_MODEL (a HF Llama-3.2-1B directory; a 2-layer model is then built
on the fly, which takes a few minutes on CPU)."""

import os

import pytest
import torch


def _const(kind):
    path = os.environ.get(f"BITEXACT_{kind.upper()}_CONST")
    if path:
        return torch.load(path, map_location="cpu", weights_only=False)
    model = os.environ.get("BITEXACT_MODEL")
    if not model:
        pytest.skip(f"set BITEXACT_{kind.upper()}_CONST or BITEXACT_MODEL")
    if kind == "exact":
        from bitexact.exact.consts import build
    else:
        from bitexact.int8.consts import build
    return build(model, n_layers=2)


@pytest.fixture(scope="session", params=["exact", "int8"])
def const(request):
    return request.param, _const(request.param)
