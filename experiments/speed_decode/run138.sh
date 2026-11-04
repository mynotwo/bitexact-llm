#!/bin/bash
# speed_decode runs on 138. Usage:
#   run138.sh gate TAG G1 G2 G3 G4   sampler gate: T1/T0 x max_num_seqs 1/64, 32 GSM8K prompts x G8 (parallel, one GPU each)
#   run138.sh speed TAG G            ms/step and tok/s at B = 1 and 64 (median of 3), sequential on one GPU
# Every GPU must show 0 MiB. Outputs: $O/TAG_{t1,t0}_ms{1,64}.json, $O/TAG_speed_B{1,64}.json
. $HOME/speed_decode/env138.sh; cd $P
MODE=$1; TAG=$2; shift 2
chk() { for g in "$@"; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" = 0 ] || { echo "GPU $g busy ($u MiB)"; exit 1; }; done; }
chk "$@"
case $MODE in
gate)
  WL="--data $DATA --n-prompts 32 --g 8"
  # one GPU per config, in the order t1/ms1, t1/ms64, t0/ms1, t0/ms64; CFGS="1 2" runs only those, GPUs given in that order
  i=0; j=0; for cfg in "t1 1.0 1" "t1 1.0 64" "t0 0 1" "t0 0 64"; do read n t ms <<< "$cfg"; i=$((i+1))
    case " ${CFGS:-1 2 3 4} " in *" $i "*) ;; *) continue;; esac
    j=$((j+1)); g=${!j}; out=$O/${TAG}_${n}_ms$ms.json
    CUDA_VISIBLE_DEVICES=$g $PY $X/sample.py --engine bitexact --model $INT $WL --temp $t --max-seqs $ms --out $out > $out.log 2>&1 &
  done; wait; ls $O/${TAG}_*_ms*.json ;;
speed)
  export CUDA_VISIBLE_DEVICES=$1
  for B in 1 64; do $PY $X/speed.py --model $INT --data $DATA --B $B --gen 512 --out $O/${TAG}_speed_B$B.json > $O/${TAG}_speed_B$B.log 2>&1; grep SPEED $O/${TAG}_speed_B$B.log; done ;;
esac
