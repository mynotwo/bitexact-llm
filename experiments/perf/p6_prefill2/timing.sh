#!/usr/bin/env bash
# p6_prefill2 timing on 141 GPU 7: A, D, C_plugin on base (25ed90e) and on the branch; 20-step pilots, seed 1, REPS reps.
S=/data41/audreymao/p6_prefill2
source /data41/audreymao/abc_rl/env.sh
export TMPDIR=$S/tmp
GPU=7
for rep in $(seq 1 ${REPS:-2}); do
  for cfg in A D Cp_base Cp_new; do
    arm=${cfg%%_*}; tree=repo; [ $cfg = Cp_base ] && tree=base; [ $arm = Cp ] && arm=C_plugin
    export PYTHONPATH=$S/$tree/src TRITON_CACHE_DIR=$S/cache/triton_$tree TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor_$tree VLLM_CACHE_ROOT=$S/cache/vllm_$tree
    cd $S/$tree/experiments/abc_rl
    while [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" != 0 ]; do sleep 20; done
    echo "$(date +%T) rep$rep $cfg gpu$GPU=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') uptime=$(uptime | sed 's/.*load average: //')" >> $S/out/timing_launch.log
    O=$S/out/t${rep}_$cfg; mkdir -p $O
    CUDA_VISIBLE_DEVICES=$GPU $PY train.py --arm $arm --seed 1 --steps 20 --eval-every 1000 --eval-n 16 --hf $HF --const $CONST --data $DATA --work $S/work --out $O > $O.log 2>&1
    echo "$(date +%T) rep$rep $cfg rc=$?" >> $S/out/timing_launch.log
  done
done
echo ALLDONE >> $S/out/timing_launch.log
