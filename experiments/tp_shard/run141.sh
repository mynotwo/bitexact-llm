#!/bin/bash
# TP-shard identity on 141 (8x H800). Usage: run141.sh [world sizes...]  default "1 2 4"
# Each world size runs in its own torchrun; GPUs are picked as the first `world` idle devices.
set -u
W=/data41/audreymao/bitexact_port
SYNC=${SYNC:-$W/tp_shard}
OUT=$SYNC/out
. $W/env.sh >/dev/null 2>&1; C=$BITEXACT_INT8_CONST
export PYTHONPATH=$SYNC/repo/src:$SYNC/code/experiments/tp_shard
mkdir -p $OUT
WORLDS=${*:-1 2 4}
for n in $WORLDS; do
  gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | head -$n | paste -sd, -)
  cnt=$(echo "$gpus" | tr ',' '\n' | grep -c .)
  [ "$cnt" -lt "$n" ] && { echo "NOT ENOUGH GPUS for world $n (have $cnt)"; continue; }
  echo "=== world $n on [$gpus] $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpus timeout 3600 $PY -m torch.distributed.run --nproc_per_node=$n --master_port=$((29500+n)) --standalone \
    $SYNC/code/experiments/tp_shard/tp_run.py --const $C --out $OUT --max-new ${MAXNEW:-64} $([ "$n" = 1 ] && echo --reference) \
    > $OUT/tp$n.log 2>&1 && echo "OK tp$n $(date +%T)" || echo "FAIL tp$n: $(tail -3 $OUT/tp$n.log | cut -c1-200)"
done
echo "=== compare"
$PY $SYNC/code/experiments/tp_shard/compare_tp.py $OUT/tp1.json $OUT/tp2.json $OUT/tp4.json 2>&1 | tail -20
