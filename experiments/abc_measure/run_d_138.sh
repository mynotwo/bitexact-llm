#!/usr/bin/env bash
# Cross-hardware check for arm D: score the H800 rollouts of arm D on an H100 (host 138) with the same vLLM
# batch-invariant engine (VLLM_BATCH_INVARIANT=1, TRITON_ATTN, bf16, scorer max_num_seqs 128, shuffled order).
# Waits for a free GPU (< 1 GB used) and for the copied inputs; never touches a busy card.
W=$HOME/abc_measure; O=$W/out; PY=$W/venv/bin/python; C=$W/code/experiments/abc_measure
export TMPDIR=$W/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_BATCH_INVARIANT=1 \
       XDG_CACHE_HOME=$W/cache TRITON_CACHE_DIR=$W/cache/triton VLLM_CACHE_ROOT=$W/cache/vllm
mkdir -p $W/cache
until $PY -c "import json,sys;json.load(open('$O/q2_16384_D.json'))" 2>/dev/null && [ -f $W/deqmodel/model.safetensors ] && ! pgrep -f "tar xf -" >/dev/null; do sleep 30; done
echo "inputs ready $(date +%T)"
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1; exit}')
  [ -n "$g" ] && break; sleep 60
done
echo "GPU $g free $(date +%T)"
export CUDA_VISIBLE_DEVICES=$g
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
for n in q1_D q2_256_D q2_1024_D q2_4096_D q2_16384_D; do
  [ -f $O/h100_$n.json ] || $PY $C/score_vllm.py --attn TRITON_ATTN --rollout $O/$n.json --model $W/deqmodel --data $W/gsm8k --max-seqs 128 --out $O/h100_$n.json || echo "FAIL $n"
done
echo ALLDONE
