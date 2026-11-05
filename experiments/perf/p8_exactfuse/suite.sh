#!/usr/bin/env bash
# Invariance suite on 141 with the exact constants. usage: bash suite.sh TAG "GPUS"   (runs detached; log $S/suite_TAG.log)
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf8_20261105/exactfuse
T=$1; G=$2
ssh $HOST ". /data41/audreymao/a17_vllm/env.sh; cd $S/repo; export PYTHONPATH=$S/repo/src:$S/repo/tests/vllm VLLM_ENABLE_V1_MULTIPROCESSING=0; nohup env GPUS='$G' PY=/data41/audreymao/a17_vllm/venv/bin/python MODEL=/data41/audreymao/perf7_20261105/exactplugin/exactmodel CONST=/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt OUT=$S/runs bash tests/vllm/run_tests.sh $T > $S/suite_$T.log 2>&1 < /dev/null & echo started"
