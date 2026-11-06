
## decode (ms per step)

| run | B | tok/s | wall | prof_wall | kernel_sum | busy | wall-busy | prof_idle | gemm_int8 | gemm_bf16 | elementwise | attention | kv_write | sampling_logits | memcpy | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_1b | 128 | 9351.6 | 8.359 | 7.841 | 8.138 | 8.136 | 0.223 | 0.181 | 4.917 | 0.000 | 1.042 | 0.741 | 0.000 | 1.236 | 0.007 | 0.196 |
| base_8b | 128 | 1985.3 | 38.551 | 35.532 | 37.539 | 37.537 | 1.014 | 0.284 | 28.505 | 0.000 | 3.478 | 2.863 | 0.000 | 2.092 | 0.006 | 0.595 |
| repo_1b | 128 | 9524.7 | 8.135 | 7.654 | 7.908 | 7.906 | 0.229 | 0.188 | 4.728 | 0.000 | 1.041 | 0.742 | 0.000 | 1.196 | 0.006 | 0.195 |
| repo_8b | 128 | 2253.6 | 30.793 | 28.425 | 29.864 | 29.863 | 0.930 | 0.356 | 2.314 | 16.190 | 2.087 | 2.880 | 0.000 | 1.682 | 0.006 | 4.706 |

## prefill (ms per step)

| run | B | tok/s | wall | prof_wall | kernel_sum | busy | wall-busy | prof_idle | gemm_int8 | gemm_bf16 | elementwise | attention | kv_write | sampling_logits | memcpy | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_1b | 128 | 9351.6 | 12.519 | 14.132 | 667.515 | 667.515 | -654.996 | 2.696 | 539.591 | 0.000 | 78.695 | 36.603 | 0.000 | 1.245 | 0.019 | 11.362 |
| base_8b | 128 | 1985.3 | 18.800 | 21.632 | 3375.690 | 3375.688 | -3356.888 | 3.014 | 2818.185 | 0.000 | 340.442 | 156.802 | 0.000 | 2.130 | 0.018 | 58.112 |
| repo_1b | 128 | 9524.7 | 12.689 | 14.016 | 666.482 | 666.479 | -653.790 | 2.614 | 538.754 | 0.000 | 78.667 | 36.499 | 0.000 | 1.202 | 0.018 | 11.341 |
| repo_8b | 128 | 2253.6 | 18.611 | 21.766 | 3371.345 | 3371.343 | -3352.732 | 3.005 | 2814.296 | 0.000 | 340.028 | 157.030 | 0.000 | 1.740 | 0.018 | 58.234 |

### top kernels: base_1b B=128 decode

- 4.917 ms x64.0 [gemm_int8] _lin_kernel
- 1.036 ms x1.0 [sampling_logits] _lin_kernel
- 0.738 ms x16.0 [attention] _xacc_kernel
- 0.572 ms x16.0 [elementwise] _silu_kernel
- 0.420 ms x33.0 [elementwise] _norm_kernel
- 0.104 ms x16.0 [other] _attn_out_kernel
- 0.075 ms x16.0 [other] _qkv_kernel
- 0.074 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.069 ms x1.0 [sampling_logits] _part_kernel
- 0.027 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.007 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel
- 0.007 ms x1.0 [other] _combine_sampled_and_draft_tokens_kernel
- 0.006 ms x1.0 [elementwise] _gather_block_tables_kernel

### top kernels: base_1b B=128 prefill

- 539.591 ms x64.0 [gemm_int8] _lin_kernel
- 46.957 ms x16.0 [elementwise] _silu_kernel
- 36.601 ms x16.0 [attention] _xacc_kernel
- 31.652 ms x33.0 [elementwise] _norm_kernel
- 6.460 ms x16.0 [other] _attn_out_kernel
- 4.846 ms x16.0 [other] _qkv_kernel
- 1.035 ms x1.0 [sampling_logits] _lin_kernel
- 0.079 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.069 ms x1.0 [sampling_logits] _part_kernel
- 0.055 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.045 ms x4.0 [other] _apply_write_kernel
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.009 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.008 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.007 ms x1.0 [sampling_logits] _post_update_kernel

### top kernels: base_8b B=128 decode

- 28.505 ms x128.0 [gemm_int8] _lin_kernel
- 2.860 ms x32.0 [attention] _xacc_kernel
- 1.923 ms x1.0 [sampling_logits] _lin_kernel
- 1.911 ms x32.0 [elementwise] _silu_kernel
- 1.492 ms x65.0 [elementwise] _norm_kernel
- 0.374 ms x32.0 [other] _attn_out_kernel
- 0.205 ms x32.0 [other] _qkv_kernel
- 0.067 ms x1.0 [sampling_logits] _part_kernel
- 0.053 ms x32.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.036 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.021 ms x1.0 [sampling_logits] _norm_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.006 ms x1.0 [sampling_logits] void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int,
- 0.006 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel
- 0.005 ms x1.0 [other] _combine_sampled_and_draft_tokens_kernel

### top kernels: base_8b B=128 prefill

- 2818.185 ms x128.0 [gemm_int8] _lin_kernel
- 171.601 ms x65.0 [elementwise] _norm_kernel
- 168.699 ms x32.0 [elementwise] _silu_kernel
- 156.800 ms x32.0 [attention] _xacc_kernel
- 41.932 ms x32.0 [other] _attn_out_kernel
- 16.139 ms x32.0 [other] _qkv_kernel
- 1.928 ms x1.0 [sampling_logits] _lin_kernel
- 0.114 ms x32.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.066 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.066 ms x1.0 [sampling_logits] _part_kernel
- 0.030 ms x4.0 [other] _apply_write_kernel
- 0.022 ms x1.0 [sampling_logits] _norm_kernel
- 0.008 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.008 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.007 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel

### top kernels: repo_1b B=128 decode

- 4.728 ms x64.0 [gemm_int8] _lin_kernel
- 1.037 ms x1.0 [sampling_logits] _lin_kernel
- 0.740 ms x16.0 [attention] _xacc_kernel
- 0.574 ms x16.0 [elementwise] _silu_kernel
- 0.418 ms x33.0 [elementwise] _norm_kernel
- 0.105 ms x16.0 [other] _attn_out_kernel
- 0.075 ms x16.0 [other] _qkv_kernel
- 0.069 ms x1.0 [sampling_logits] _part_kernel
- 0.034 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.027 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.006 ms x1.0 [sampling_logits] void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int,
- 0.005 ms x1.0 [sampling_logits] _get_num_sampled_and_rejected_kernel
- 0.005 ms x1.0 [other] _combine_sampled_and_draft_tokens_kernel

### top kernels: repo_1b B=128 prefill

- 538.754 ms x64.0 [gemm_int8] _lin_kernel
- 46.938 ms x16.0 [elementwise] _silu_kernel
- 36.497 ms x16.0 [attention] _xacc_kernel
- 31.644 ms x33.0 [elementwise] _norm_kernel
- 6.457 ms x16.0 [other] _attn_out_kernel
- 4.841 ms x16.0 [other] _qkv_kernel
- 1.039 ms x1.0 [sampling_logits] _lin_kernel
- 0.068 ms x1.0 [sampling_logits] _part_kernel
- 0.057 ms x16.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.035 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.032 ms x4.0 [other] _apply_write_kernel
- 0.012 ms x1.0 [sampling_logits] _norm_kernel
- 0.008 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
- 0.008 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.006 ms x4.0 [memcpy] Memcpy DtoD (Device -> Device)

### top kernels: repo_8b B=128 decode

- 16.190 ms x96.0 [gemm_bf16] _wgmma
- 3.879 ms x96.0 [other] _epilogue
- 2.878 ms x32.0 [attention] _xacc_kernel
- 2.314 ms x32.0 [gemm_int8] _lin_kernel
- 1.197 ms x32.0 [elementwise] _silu_cached_kernel
- 1.138 ms x1.0 [sampling_logits] _wgmma
- 0.814 ms x65.0 [elementwise] _norm_kernel
- 0.389 ms x96.0 [other] _stack
- 0.383 ms x1.0 [sampling_logits] _epilogue
- 0.216 ms x32.0 [other] _attn_recip_kernel
- 0.205 ms x32.0 [other] _qkv_kernel
- 0.067 ms x1.0 [sampling_logits] _part_kernel
- 0.053 ms x32.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.034 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.011 ms x1.0 [sampling_logits] _norm_kernel

### top kernels: repo_8b B=128 prefill

- 2814.296 ms x128.0 [gemm_int8] _lin_kernel
- 171.376 ms x65.0 [elementwise] _norm_kernel
- 168.511 ms x32.0 [elementwise] _silu_kernel
- 157.028 ms x32.0 [attention] _xacc_kernel
- 42.043 ms x32.0 [other] _attn_out_kernel
- 16.148 ms x32.0 [other] _qkv_kernel
- 1.172 ms x1.0 [sampling_logits] _wgmma
- 0.384 ms x1.0 [sampling_logits] _epilogue
- 0.113 ms x32.0 [elementwise] void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_
- 0.068 ms x1.0 [sampling_logits] _part_kernel
- 0.053 ms x1.0 [sampling_logits] _gumbel_sample_kernel
- 0.032 ms x4.0 [other] _apply_write_kernel
- 0.011 ms x1.0 [sampling_logits] _norm_kernel
- 0.008 ms x1.0 [elementwise] _prepare_prefill_inputs_kernel
- 0.007 ms x3.0 [sampling_logits] Memcpy DtoH (Device -> Pinned)
