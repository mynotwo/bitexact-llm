#!/usr/bin/env bash
# run a command on 141 in the synced repo on GPU $1 (must be free): bash r.sh GPU cmd...
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf8_20261105/exactfuse
G=$1; shift
ssh $HOST ". /data41/audreymao/a17_vllm/env.sh; cd $S/${TREE:-repo}; export PYTHONPATH=$S/${TREE:-repo}/src:$S/${TREE:-repo}/tests/vllm VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=$G; $*"
