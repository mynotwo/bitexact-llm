"""Exact engine geometry beyond Llama-3.2-1B: RMSNorm row lengths, the head_dim 128 score scale, sharded
checkpoints, and a tiny GQA model with head_dim 128 (CPU) checked against HF fp32 and across batch / cache."""

import json
from decimal import Decimal, getcontext

import pytest
import torch

from bitexact.exact import ops
from bitexact.exact.consts import FR
from bitexact.llama import FF, HD, HKV, D, H, dims


@pytest.mark.parametrize("width", [512, 2048, 4096])
def test_rms_widths(width):
    g = torch.Generator().manual_seed(width)
    x = torch.randint(-(200 << FR), 200 << FR, (4, width), generator=g)
    x[1] //= 1 << 12
    x[2] = (1 << 40) - 1                                         # residual bound: sum of squares hits 2^62
    r, er = ops.rms(x)
    assert bool((r >= 1 << 30).all()) and bool((r < 1 << 31).all())
    ref = (x.double() / 2 ** FR).pow(2).mean(-1).add(1e-5).sqrt()
    assert torch.allclose(r.double() * torch.pow(2.0, er.double()), ref, rtol=1e-7)


def test_rms_rejects_other_widths():
    with pytest.raises(AssertionError):
        ops.rms(torch.ones(2, 3000, dtype=torch.long))


def test_score_scale_constants():
    getcontext().prec = 60
    l2e = 1 / Decimal(2).ln()
    assert ops.L2E28_HD[64] == ops.L2E28 == int((l2e * 2 ** 28).to_integral_value())
    # 1 / sqrt(128) = 2^-3 / sqrt(2): the 2^-3 is the same shift as for head_dim 64
    assert ops.L2E28_HD[128] == int((l2e / Decimal(2).sqrt() * 2 ** 28).to_integral_value())


def test_dims_defaults_and_8b():
    assert dims({}) == (H, HKV, HD, D, FF)
    cfg8 = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336}
    assert dims(cfg8) == (32, 8, 128, 4096, 14336)


def _tiny(tmp_path, shards):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=128, max_position_embeddings=4096,
                      rope_theta=500000.0, tie_word_embeddings=False, rms_norm_eps=1e-5,
                      rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for ly in m.model.layers:                                # larger scores: the scale must matter
            ly.self_attn.q_proj.weight.mul_(4)
            ly.self_attn.k_proj.weight.mul_(4)
    m = m.to(torch.bfloat16)
    m.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="1MB" if shards else "10GB")
    return tmp_path


def test_sharded_checkpoint(tmp_path):
    from bitexact.llama import load_checkpoint
    one, many = _tiny(tmp_path / "one", False), _tiny(tmp_path / "many", True)
    assert (many / "model.safetensors.index.json").exists()
    c1, s1 = load_checkpoint(str(one))
    c2, s2 = load_checkpoint(str(many))
    assert set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
    assert json.loads((one / "config.json").read_text())["head_dim"] == 128


def test_tiny_gqa_hd128(tmp_path):
    from transformers import LlamaForCausalLM

    from bitexact.exact.consts import build
    from bitexact.exact.model import ExactLlama
    path = _tiny(tmp_path, False)
    c = build(str(path))
    m = ExactLlama(c, "cpu")
    assert (m.H, m.HKV, m.HD, m.D) == (4, 2, 128, 512)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 512, (2, 24), generator=g)
    T = ids.shape[1]
    pos = torch.arange(T).expand(2, T).contiguous()
    with torch.no_grad():
        full = m.logits(m.forward(ids, pos, m.new_cache(2, T)).reshape(-1, m.D)).view(2, T, -1)
        one = m.logits(m.forward(ids[1:, :17], pos[1:, :17], m.new_cache(1, T)).reshape(-1, m.D))
        assert torch.equal(one, full[1, :17])                     # batch and length independent
        cache = m.new_cache(1, T)
        m.forward(ids[:1, :10], pos[:1, :10], cache)
        rows = [m.logits(m.forward(ids[:1, t:t + 1], pos[:1, t:t + 1], cache)[:, 0]) for t in range(10, T)]
        assert torch.equal(torch.cat(rows), full[0, 10:])         # KV cache
        m.check()
        hf = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager")
        ref = hf(ids).logits.double()
    err = (full.double() / 2 ** FR - ref).abs().max().item()
    assert err < 2e-4 * ref.abs().max().item(), err
