#!/usr/bin/env bash
# bash run_bitcheck.sh MODE GPU   (MODE old | new = code_old | code_new on 138); then compare:
#   $PY bitcheck.py compare runs/bitcheck_old.pt runs/bitcheck_new.pt
mode=$1
source "$(dirname "$0")/env.sh" $mode bitcheck_$mode
export CUDA_VISIBLE_DEVICES=$2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $S/work_bitcheck_$mode
$PY $S/code_$mode/experiments/abc_rl_speed/bitcheck.py run --mode $mode --hf "$HF" --const "$CONST" --data "$DATA" \
  --work $S/work_bitcheck_$mode --steps 10 --out $S/runs/bitcheck_$mode.pt > $S/runs/bitcheck_$mode.log 2>&1
echo "bitcheck_$mode exit $?" >> $S/runs/queue.log
