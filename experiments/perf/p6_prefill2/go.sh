#!/usr/bin/env bash
# usage: go.sh TREE GPU OUTNAME [prof_score args...]   (runs on 141)
S=/data41/audreymao/p6_prefill2; T=$1; G=$2; N=$3; shift 3
source /data41/audreymao/abc_rl/env.sh
export PYTHONPATH=$S/$T/src TRITON_CACHE_DIR=$S/cache/triton_$T TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor_$T VLLM_CACHE_ROOT=$S/cache/vllm_$T TMPDIR=$S/tmp
cd $S/$T/experiments/perf/p6_prefill2
CUDA_VISIBLE_DEVICES=$G $PY prof_score.py --hf $HF --const $CONST --data $DATA --work $S/work --out $S/out/$N "$@"
