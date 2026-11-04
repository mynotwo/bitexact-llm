#!/usr/bin/env bash
# Scorer gate (8 configurations) on 141 (H800): bash run141.sh GPU, from $S/code (git archive of the tested commit).
# $S/out holds copies of the 138 plugin rollouts and the eager references recorded on H100.
S=/data41/audreymao/speed_on_main_20261104; W=/data41/audreymao/abc_measure
PY=/data41/audreymao/bitexact_port/venv/bin/python; CONST=/data41/audreymao/bitexact_port/data/intllama_const.pt
export PYTHONPATH=$S/code/src:$S/code/experiments/rl_mismatch TMPDIR=$S/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 \
  XDG_CACHE_HOME=$S/cache TRITON_CACHE_DIR=$S/cache/triton TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor
u=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" = 0 ] || { echo "GPU $1 busy"; exit 1; }
export CUDA_VISIBLE_DEVICES=$1
B="$PY $S/code/experiments/speed_on_main/scorer_bench.py --const $CONST --model $W/deqmodel --data $W/gsm8k --mode packed"
O=$S/out
echo "START gpu=$1 $(nvidia-smi -i $1 --query-gpu=name,driver_version --format=csv,noheader) $(date +%T)"
for b in 8 64 1; do $B --rollout $O/q1_B.json --nseq 128 --bs $b --check --ref $O/ref_gsm.pt --tag h800-gsm-b$b; done
$B --rollout $O/q1_B.json --nseq 128 --bs 8 --order shuffle --check --ref $O/ref_gsm.pt --tag h800-gsm-b8-shuffle
for b in 4 1 8; do $B --rollout $O/q2_4096_B.json --bs $b --check --ref $O/ref_4k.pt --tag h800-4k-b$b; done
$B --rollout $O/q2_4096_B.json --bs 4 --order shuffle --check --ref $O/ref_4k.pt --tag h800-4k-b4-shuffle
echo ALLDONE
