
## decode (ms per step)

| run | B | tok/s | wall | prof_wall | kernel_sum | busy | wall-busy | prof_idle | gemm_int8 | gemm_bf16 | elementwise | attention | kv_write | sampling_logits | memcpy | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_1b | 128 | 9480.3 | 8.205 | 7.674 | 7.957 | 7.956 | 0.249 | 0.183 | 4.729 | 0.000 | 1.041 | 0.741 | 0.000 | 1.244 | 0.007 | 0.196 |
| repo_1b | 128 | 9623.9 | 7.997 | 7.501 | 7.727 | 7.725 | 0.272 | 0.208 | 2.649 | 1.487 | 1.041 | 0.776 | 0.000 | 1.206 | 0.007 | 0.561 |

## prefill (ms per step)

| run | B | tok/s | wall | prof_wall | kernel_sum | busy | wall-busy | prof_idle | gemm_int8 | gemm_bf16 | elementwise | attention | kv_write | sampling_logits | memcpy | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_1b | 128 | 9480.3 | 12.730 | 14.095 | 667.476 | 667.476 | -654.746 | 2.650 | 539.622 | 0.000 | 78.683 | 36.518 | 0.000 | 1.276 | 0.019 | 11.358 |
| repo_1b | 128 | 9623.9 | 12.375 | 13.812 | 667.415 | 667.415 | -655.040 | 2.643 | 539.631 | 0.000 | 78.696 | 36.514 | 0.000 | 1.208 | 0.019 | 11.347 |

### top kernels: base_1b B=128 decode

- 4.729 ms x64.0 [gemm_int8] _lin_kernel
- 1.046 ms x1.0 [sampling_logits] _lin_kernel
- 0.738 ms x16.0 [attention] _xacc_kernel
- 0.573 ms x16.0 [elementwise] _silu_kernel
- 0.418 ms x33.0 [elementwise] _norm_kernel
- 0.105 ms x16.0 [other] _attn_out_kernel
- 0.074 ms x16.0 [other] _qkv_kernel
- 0.072 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.069 ms x1.0 [sampling_logits] _part_kernel
- 0.026 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.007 ms x1.0 [other] _combine_sampled_and_draft_tokens_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.007 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel
- 0.006 ms x1.0 [sampling_logits] void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int,

### top kernels: base_1b B=128 prefill

- 539.622 ms x64.0 [gemm_int8] _lin_kernel
- 46.938 ms x16.0 [elementwise] _silu_kernel
- 36.516 ms x16.0 [attention] _xacc_kernel
- 31.660 ms x33.0 [elementwise] _norm_kernel
- 6.459 ms x16.0 [other] _attn_out_kernel
- 4.844 ms x16.0 [other] _qkv_kernel
- 1.054 ms x1.0 [sampling_logits] _lin_kernel
- 0.095 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.070 ms x1.0 [sampling_logits] _part_kernel
- 0.055 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.044 ms x4.0 [other] _apply_write_kernel
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.009 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.007 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel

### top kernels: repo_1b B=128 decode

- 2.649 ms x32.0 [gemm_int8] _lin_kernel
- 1.487 ms x32.0 [gemm_bf16] _wgmma
- 1.050 ms x1.0 [sampling_logits] _lin_kernel
- 0.773 ms x16.0 [attention] _xacc_kernel
- 0.572 ms x16.0 [elementwise] _silu_kernel
- 0.421 ms x33.0 [elementwise] _norm_kernel
- 0.260 ms x32.0 [other] _epilogue
- 0.106 ms x32.0 [other] _stack
- 0.105 ms x16.0 [other] _attn_out_kernel
- 0.075 ms x16.0 [other] _qkv_kernel
- 0.068 ms x1.0 [sampling_logits] _part_kernel
- 0.032 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.026 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.013 ms x1.0 [sampling_logits] _norm_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)

### top kernels: repo_1b B=128 prefill

- 539.631 ms x64.0 [gemm_int8] _lin_kernel
- 46.937 ms x16.0 [elementwise] _silu_kernel
- 36.512 ms x16.0 [attention] _xacc_kernel
- 31.677 ms x33.0 [elementwise] _norm_kernel
- 6.459 ms x16.0 [other] _attn_out_kernel
- 4.846 ms x16.0 [other] _qkv_kernel
- 1.049 ms x1.0 [sampling_logits] _lin_kernel
- 0.069 ms x1.0 [sampling_logits] _part_kernel
- 0.056 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.032 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.032 ms x4.0 [other] _apply_write_kernel
- 0.013 ms x1.0 [sampling_logits] _norm_kernel
- 0.008 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.008 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.006 ms x4.0 [memcpy] Memcpy DtoD (Device -> Device)
