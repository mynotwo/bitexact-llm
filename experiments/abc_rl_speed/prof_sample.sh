#!/usr/bin/env bash
# usage: prof_sample.sh GPU tag ARM GPU_MEM [extra args]   (output: ~/abc_rl_speed/runs/ps_<tag>.log)
gpu=$1; tag=$2; arm=$3; mem=$4; shift 4
source ~/abc_rl_speed/code_new/experiments/abc_rl_speed/env.sh new ps_$tag
export CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ~/abc_rl_speed/work_ps_$tag
$PY $C/../abc_rl_speed/prof_sample.py --arm $arm --gpu-mem $mem --hf "$HF" --const "$CONST" --data "$DATA" --work ~/abc_rl_speed/work_ps_$tag "$@" > ~/abc_rl_speed/runs/ps_$tag.log 2>&1
echo "ALLDONE $?" >> ~/abc_rl_speed/runs/ps_$tag.log
