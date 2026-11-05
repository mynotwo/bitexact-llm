#!/usr/bin/env bash
# check_plug.py (RL scoring path) on the synced tree. usage: bash check_plug.sh GPU [TAG]   (run on 141)
S=/data41/audreymao/perf6_20261105/graphlogits
GPU=${1:?gpu}; TAG=${2:-gl}
source /data41/audreymao/abc_rl/env.sh
mkdir -p $S/cache/triton $S/cache/inductor $S/cache/vllm $S/tmp $S/out/cp_$TAG
export PYTHONPATH=$S/repo/src TRITON_CACHE_DIR=$S/cache/triton TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor VLLM_CACHE_ROOT=$S/cache/vllm TMPDIR=$S/tmp
cd $S/repo/experiments/abc_rl
CUDA_VISIBLE_DEVICES=$GPU $PY check_plug.py --hf $HF --const $CONST --data $DATA --work $S/work_$TAG --out $S/out/cp_$TAG/res > $S/out/cp_$TAG/check_plug.log 2>&1
echo rc=$? >> $S/out/cp_$TAG/done
