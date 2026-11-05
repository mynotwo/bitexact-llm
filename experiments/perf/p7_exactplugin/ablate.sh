#!/usr/bin/env bash
# exact plugin: fused linear vs original wmm_triton + lin_y path (BITEXACT_EXACT_FUSE_M=0), interleaved, 3 reps. GPU=<g> bash ablate.sh OUT
set -u
S=/data41/audreymao/perf7_20261105/exactplugin; PY=/data41/audreymao/a17_vllm/venv/bin/python
. /data41/audreymao/a17_vllm/env.sh
O=$1; mkdir -p $O; cd $S/${TREE:-repo}
export PYTHONPATH=$S/${TREE:-repo}/src:$S/${TREE:-repo}/tests/vllm VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=$GPU
busy() { [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; }
for rep in 1 2 3; do for b in 1 32 128; do for cfg in fused unfused; do
  while busy; do sleep 30; done; sleep 3
  fm=$([ $cfg = fused ] && echo 1073741824 || echo 0)
  BITEXACT_EXACT_FUSE_M=$fm timeout 1200 $PY tests/vllm/bench.py --model $S/exactmodel --batch $b --tag ${cfg}_b${b}_r$rep > $O/${cfg}_b${b}_r$rep.log 2>&1
  busy && echo "CONTAMINATED ${cfg}_b${b}_r$rep" >> $O/progress.log
done; done; done
echo ALLDONE >> $O/progress.log
