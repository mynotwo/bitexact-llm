#!/usr/bin/env bash
# Micro benchmark (micro.py) of several synced repo copies on one GPU, one after another (141).
#   bash micro_set.sh OUTTAG "VARIANTS" "ARGSETS"
#   e.g. bash micro_set.sh r1 "base ab" "--batch=128 --batch=128,--dyn --batch=32,--dyn"
# VARIANTS are directories under $S holding a repo copy; each ARGSET is micro.py arguments with ',' for spaces.
# The GPU is the first of $GPUS that is free (0 MiB, no compute process); before every run it must be free
# again, else the script waits. Outputs: $S/out/OUTTAG_<variant>_<argset>.{log,json}, summary on stdout.
set -u
S=/data41/audreymao/speedup_20261030/fuse
TAG=$1; VARS=$2; SETS=$3
GPUS=${GPUS:-0 1 7}
. /data41/audreymao/a17_vllm/env.sh
PY=/data41/audreymao/a17_vllm/venv/bin/python
export BITEXACT_INT8_CONST=/data41/audreymao/a17_vllm/data/intllama_const.pt
export TORCHINDUCTOR_CACHE_DIR=$S/inductor_cache          # own cache: variants must not share compiled kernels
isfree() { m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  p=$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$m" = 0 ] && [ "$p" = 0 ]; }
G=""
while [ -z "$G" ]; do
  for g in $GPUS; do case $g in 2|3|4|5|6) continue;; esac; isfree $g && { G=$g; break; }; done
  [ -z "$G" ] && sleep 20
done
echo "GPU $G"
for a in $SETS; do
  for v in $VARS; do
    n=0; while ! isfree $G; do [ $((n % 15)) = 0 ] && echo "wait GPU $G"; n=$((n + 1)); sleep 20; done
    name=${TAG}_${v}_$(echo $a | tr -d '-' | tr ',=' '__')
    args=$(echo $a | tr ',' ' ')
    CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$S/$v/src timeout 1800 $PY $S/$v/experiments/perf/fuse/micro.py $args \
      --out $S/out/$name.json > $S/out/$name.log 2>&1
    echo "$name rc=$? $(grep -h '^RESULT' $S/out/$name.log | cut -c8-400)"
  done
done
echo ALLDONE
