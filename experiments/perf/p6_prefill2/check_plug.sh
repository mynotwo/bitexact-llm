#!/usr/bin/env bash
# check_plug.py on the merged tree, fused prefill ON (default), GPU 0
S=/data41/audreymao/p6_prefill2
source /data41/audreymao/abc_rl/env.sh
export PYTHONPATH=$S/repo/src TRITON_CACHE_DIR=$S/cache/triton TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor VLLM_CACHE_ROOT=$S/cache/vllm TMPDIR=$S/tmp BITEXACT_PREFILL_FUSE=1
cd $S/repo/experiments/abc_rl
CUDA_VISIBLE_DEVICES=0 $PY check_plug.py --hf $HF --const $CONST --data $DATA --work $S/work --out $S/out/check_plug > $S/out/check_plug.log 2>&1
echo rc=$? >> $S/out/check_plug.done
