#!/bin/bash
# Throughput + identity for the sharded engine on 141. Usage: bench.sh "<worlds>" (default "1 2 4")
set -u
W=/data41/audreymao/bitexact_port
. $W/env.sh >/dev/null 2>&1; C=$BITEXACT_INT8_CONST
S=$W/tp_shard/code/experiments/tp_shard
export PYTHONPATH=$W/tp_shard/repo/src:$S
mkdir -p $W/tp_shard/out
for n in ${1:-1 2 4}; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | head -$n | paste -sd, -)
  echo "=== world $n on $GPU $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU timeout 900 $PY -m torch.distributed.run --nproc_per_node=$n --master_port=$((29640+n)) --standalone \
    $S/bench_tp.py --const $C --bs 1 8 32 --steps 16 --out $W/tp_shard/out/bench_tp$n.json \
    2>&1 | grep -E '"world"|Error' | head -5
done
echo "=== summary"
$PY - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob("/data41/audreymao/bitexact_port/tp_shard/out/bench_tp*.json")):
    d = json.load(open(f))
    rows += [(r["world"], r["batch"], r["ms_step"], r["tok_s"]) for r in d["rows"]]
print(f"{'TP':4} {'batch':6} {'ms/step':9} {'tok/s':9}")
for tp, b, ms, tk in sorted(rows):
    print(f"{tp:<4} {b:<6} {ms:<9.2f} {tk:<9.1f}")
PY
