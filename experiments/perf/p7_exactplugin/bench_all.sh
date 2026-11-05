#!/usr/bin/env bash
# Interleaved e2e bench on ONE GPU (runs on 141): for rep, for batch, for config: tests/vllm/bench.py (128 in / 128 out, T=0).
# usage (on 141): GPU=2 REPS=5 BATCHES="1 32 128" CONFIGS="exact int8 bf16 bi" bash bench_all.sh OUTDIR
set -u
S=/data41/audreymao/perf7_20261105/exactplugin; PY=/data41/audreymao/a17_vllm/venv/bin/python
. /data41/audreymao/a17_vllm/env.sh
D=/data41/audreymao/perf_20261030
O=$1; mkdir -p $O
cd $S/${TREE:-repo}
export PYTHONPATH=$S/${TREE:-repo}/src:$S/${TREE:-repo}/tests/vllm VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=$GPU
for rep in $(seq 1 ${REPS:-5}); do
  for b in ${BATCHES:-1 32 128}; do
    for cfg in ${CONFIGS:-exact int8 bf16 bi}; do
      case $cfg in
        exact) cmd="$PY tests/vllm/bench.py --model $S/exactmodel";;
        int8)  cmd="$PY tests/vllm/bench.py --model $D/det_compare/intmodel";;
        bf16)  cmd="$PY tests/vllm/bench.py --model $D/bf16model";;
        bi)    cmd="env VLLM_BATCH_INVARIANT=1 $PY tests/vllm/bench.py --model $D/bf16model --attn TRITON_ATTN";;
      esac
      echo "$(date +%T) $cfg b$b r$rep gpu=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)" >> $O/progress.log
      timeout 1200 $cmd --batch $b --tag ${cfg}_b${b}_r$rep > $O/${cfg}_b${b}_r$rep.log 2>&1
    done
  done
done
echo ALLDONE >> $O/progress.log
