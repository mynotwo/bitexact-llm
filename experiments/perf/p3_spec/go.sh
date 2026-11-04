#!/usr/bin/env bash
# on 141: bash go.sh GPU "cmd args..."  (runs spec_check.py from $S/repo with the plugin env, waits for free GPU)
. /data41/audreymao/a17_vllm/env.sh
S=/data41/audreymao/p3_spec; PY=/data41/audreymao/a17_vllm/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
export VLLM_ENABLE_V1_MULTIPROCESSING=0
GPU=$1; shift; TREE=$1; shift
case " 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
while [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" != 0 ] || \
      [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; do sleep 20; done
cd $S/$TREE && PYTHONPATH=$S/$TREE/src:$S/$TREE/tests/vllm CUDA_VISIBLE_DEVICES=$GPU $PY tests/vllm/spec_check.py --model $MODEL "$@"
