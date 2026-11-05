#!/usr/bin/env bash
# 8B e2e (exact vs int8 plugin), interleaved, one GPU. GPU=3 REPS=3 bash bench8.sh OUT  (on 141)
set -u
S=/data41/audreymao/perf8_20261105/exactfuse; PY=/data41/audreymao/a17_vllm/venv/bin/python
. /data41/audreymao/a17_vllm/env.sh
O=$1; mkdir -p $O; cd $S/${TREE:-repo}
export PYTHONPATH=$S/${TREE:-repo}/src:$S/${TREE:-repo}/tests/vllm VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=$GPU
busy() { [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; }
for rep in $(seq 1 ${REPS:-3}); do for b in 1 32 128; do for cfg in exact int8; do
  case $cfg in exact) M=/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel;; int8) M=/data41/audreymao/p3_8b/intmodel8b;; esac
  while busy; do sleep 30; done; sleep 3
  echo "$(date +%T) $cfg b$b r$rep" >> $O/progress.log
  GPU_UTIL=0.6 timeout 1800 $PY tests/vllm/bench.py --model $M --batch $b --tag ${cfg}8_b${b}_r$rep > $O/${cfg}8_b${b}_r$rep.log 2>&1
  busy && echo "CONTAMINATED ${cfg}_b${b}_r$rep" >> $O/progress.log
done; done; done
echo ALLDONE >> $O/progress.log
