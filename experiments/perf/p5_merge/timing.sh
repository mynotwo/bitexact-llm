#!/usr/bin/env bash
# p5_merge timing: A, D, C_plugin (fuse off), C_plugin (fuse on); 20-step pilots, GPU 7, sequential, REPS reps.
S=/data41/audreymao/p5_merge
source /data41/audreymao/abc_rl/env.sh
export PYTHONPATH=$S/repo/src TRITON_CACHE_DIR=$S/cache/triton TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor VLLM_CACHE_ROOT=$S/cache/vllm TMPDIR=$S/tmp
C=$S/repo/experiments/abc_rl; cd $C
GPU=7
for rep in $(seq 1 ${REPS:-2}); do
  for cfg in A D Cp_off Cp_on; do
    arm=${cfg%%_*}; fuse=1; [ $cfg = Cp_off ] && fuse=0; [ $arm = Cp ] && arm=C_plugin
    while [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" != 0 ]; do sleep 20; done
    echo "$(date +%T) rep$rep $cfg gpu$GPU=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') uptime=$(uptime | sed 's/.*load average: //')" >> $S/out/timing_launch.log
    O=$S/out/r${rep}_$cfg
    CUDA_VISIBLE_DEVICES=$GPU BITEXACT_PREFILL_FUSE=$fuse $PY train.py --arm $arm --seed 1 --steps 20 --eval-every 1000 --eval-n 16 --hf $HF --const $CONST --data $DATA --work $S/work --out $O > $O.log 2>&1
    echo "$(date +%T) rep$rep $cfg rc=$?" >> $S/out/timing_launch.log
  done
done
echo ALLDONE >> $S/out/timing_launch.log
