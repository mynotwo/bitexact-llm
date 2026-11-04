#!/usr/bin/env bash
# One timed RL run on one GPU: bash run_speed.sh ARM IMPL GPU STEPS [SEED]
#   ARM IMPL: C old | C new | D old.  old = code of exp/abc-rl (src), new = code of exp/abc-rl-speed.
# Seed 1, all other settings are train.py's defaults (the settings of the RL runs); no evaluation.
arm=$1; impl=$2; gpu=$3; steps=$4; seed=${5:-1}
tag=${arm}_${impl}${TAGX:-}; code=$impl
source "$(dirname "$0")/env.sh" $code $tag
[ "$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)" = 0 ] || { echo "GPU $gpu busy"; exit 1; }
out=$S/runs/${tag}_s$seed; rm -rf $out; mkdir -p $S/work_$tag
extra="${EXTRA:-}"; [ "$impl" = new ] && extra="$extra --int-impl new"
CUDA_VISIBLE_DEVICES=$gpu $PY $C/train.py --arm $arm --seed $seed --steps $steps --eval-every 0 $extra \
  --hf "$HF" --const "$CONST" --data "$DATA" --work $S/work_$tag --out $out > $out.log 2>&1
echo "$tag exit $? $(date)" >> $S/runs/queue.log
