#!/usr/bin/env bash
# on 141: wait for a free GPU among 0-3 (<50 MiB, no compute process), then run the given command with GPU=<it>.
# usage: bash launch_free.sh cmd...      (the command reads $GPU)
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 50 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
while true; do for g in 0 1 2 3; do if free $g; then sleep 3; free $g && { export GPU=$g; echo "gpu $g"; exec "$@"; }; fi; done; sleep 30; done
