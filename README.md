# BitExact LLM

Bitwise-reproducible LLM inference and neural compression across hardware.

Personal research project by Yu (Audrey) Mao.

## Source release

The implementation and reproducibility artifacts are being prepared for release. This repository currently contains the project overview only.

## Project scope

- Integer-exact Llama inference using int8 tensor-core matrix multiplication and integer or lookup-table nonlinear operations.
- Batch-invariant Triton inference with fixed reduction and attention traversal order.
- Integration with vLLM paged KV cache and deterministic sampling.
- GPU neural compression with an integer predictor and rANS entropy coding.
- Lossless compression of agent trajectories using reproducible model predictions.

The release will document tested hardware, software versions, model configurations, and benchmark conditions. Reproducibility claims are limited to tested configurations.

Homepage: https://mynotwo.github.io/
