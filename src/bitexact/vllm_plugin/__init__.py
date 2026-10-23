"""Out-of-tree vLLM plugin serving the int8 engine with schedule-independent outputs.

Entry points (pyproject.toml): `vllm.general_plugins: bitexact = bitexact.vllm_plugin:register` registers
the model class under ARCH in every vLLM process; `vllm.logits_processors: bitexact_sampler =
bitexact.vllm_plugin.sampler:IntSampler` forces vLLM's sampler onto the integer-sampled token.
A model directory for vLLM is made with `python -m bitexact.vllm_plugin.make_model_dir`.
"""

ARCH = "BitExactLlamaForCausalLM"


def register():
    from vllm import ModelRegistry
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "bitexact.vllm_plugin.model:BitExactLlamaForCausalLM")
