. /data00/tmp/audreymao/rl_mismatch/env.sh
X=""; [ -f $O/g_bx_h100_bs16.json ] && X="bx_h100_standalone_bs16=$O/b0_bx.json,$O/g_bx_h100_bs16.json"
$PY $C/analyze.py --tokenizer $BF16 --data $DATA --out $O/metrics.json \
 --mismatch vllm_vs_hf_bf16_bs8=$O/r0_bf16.json,$O/s_hf_bf16_bs8.json vllm_vs_hf_bf16_bs1=$O/r0_bf16.json,$O/s_hf_bf16_bs1.json \
   vllm_vs_hf_bf16_bs32=$O/r0_bf16.json,$O/s_hf_bf16_bs32.json vllm_vs_hf_fp32_bs8=$O/r0_bf16.json,$O/s_hf_fp32_bs8.json \
   hf_bs1_vs_hf_bs8=$O/s_hf_bf16_bs1.json,$O/s_hf_bf16_bs8.json,$O/r0_bf16.json hf_bs1_vs_hf_bs32=$O/s_hf_bf16_bs1.json,$O/s_hf_bf16_bs32.json,$O/r0_bf16.json \
   hf_bf16bs8_vs_fp32=$O/s_hf_bf16_bs8.json,$O/s_hf_fp32_bs8.json,$O/r0_bf16.json \
   bx_plugin_vs_h800_bs8=$O/b0_bx.json,$O/s_bx_h800_bs8.json bx_plugin_vs_h800_bs32=$O/b0_bx.json,$O/s_bx_h800_bs32.json \
   bx_plugin_vs_h100_bs16=$O/b0_bx.json,$O/s_bx_h100_bs16.json bx_plugin_vs_cpu=$O/b0_bx.json,$O/s_bx_cpu.json \
 --replay vllm_repeat_same_cfg=$O/r0_bf16.json,$O/r0b_bf16_repeat.json vllm_shuffle_ms32=$O/r0_bf16.json,$O/r1_bf16_shuf32.json \
   vllm_ms8=$O/r0_bf16.json,$O/r2_bf16_seq8.json vllm_h100=$O/r0_bf16.json,$O/r3_bf16_h100.json \
   bx_plugin_shuffle_ms32=$O/b0_bx.json,$O/b1_bx_shuf32.json bx_plugin_h100=$O/b0_bx.json,$O/b2_bx_h100.json \
   bx_h800_standalone_bs64=$O/b0_bx.json,$O/g_bx_h800_bs64.json bx_cpu_standalone=$O/b0_bx.json,$O/g_bx_cpu.json $X 2>&1 | grep -v transformers
