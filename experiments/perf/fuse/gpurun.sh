#!/usr/bin/env bash
# Run one command on the first free GPU of $GPUS (default "0 1 7"; 2-6 are never used) on 141.
#   GPUS="0 1 7" TMO=1800 MAXWAIT=14400 bash gpurun.sh LOG CMD...
# Free = memory.used 0 MiB and no compute process, checked right before launch. Prints the GPU used.
set -u
LOG=$1; shift
GPUS=${GPUS:-0 1 7}; TMO=${TMO:-1800}; MAXWAIT=${MAXWAIT:-14400}
P0=${PYTHONPATH:-}; . /data41/audreymao/a17_vllm/env.sh; export PYTHONPATH=$P0   # env.sh points PYTHONPATH elsewhere
free_gpu() {
  for g in $GPUS; do
    case $g in 2|3|4|5|6) continue ;; esac
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    p=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [ "$m" = 0 ] && [ "$p" = 0 ] && { echo $g; return; }
  done; }
w=0
while g=$(free_gpu); [ -z "$g" ]; do
  [ $w -ge $MAXWAIT ] && { echo "FAIL no free GPU after ${MAXWAIT}s"; exit 9; }
  sleep 20; w=$((w + 20)); done
echo "$(date +%T) GPU $g: $*"
CUDA_VISIBLE_DEVICES=$g timeout $TMO "$@" > $LOG 2>&1; rc=$?
echo "$(date +%T) rc=$rc $(grep -h '^RESULT\|passed\|failed' $LOG | tail -2 | cut -c1-300)"
exit $rc
