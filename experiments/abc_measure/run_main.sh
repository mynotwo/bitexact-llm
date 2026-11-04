#!/usr/bin/env bash
# Q1 + Q2 measurements on ONE GPU (set CUDA_VISIBLE_DEVICES). Needs env.sh (W O C PY CONST ORIG DEQ INT_ORIG DATA).
# Resumable: a step is skipped when its output exists.
. /data41/audreymao/abc_measure/env.sh
cd $W; mkdir -p $O
step() { out=$1; shift; if [ -f "$out" ]; then echo "SKIP $out"; return; fi; echo "RUN $out $(date +%T)"; "$@" || echo "FAIL $out"; }
roll() { name=$1; eng=$2; model=$3; shift 3; step $O/$name.json $PY $C/rollout.py --engine $eng --model $model --data $DATA --out $O/$name.json "$@"; }
sc() { name=$1; mode=$2; roll=$3; model=$4; shift 4; step $O/s_$name.json $PY $C/score.py --mode $mode --rollout $O/$roll.json --model $model --const $CONST --data $DATA --out $O/s_$name.json "$@"; }

# ---- Q1: 128 prompts x G 8, <= 256 tokens, stop at EOS ----
roll q1_Aorig bf16 $ORIG
roll q1_A bf16 $DEQ
roll q1_B bitexact $INT_ORIG
sc q1_Aorig_hfbf16 hf_bf16 q1_Aorig $ORIG --bs 8
sc q1_Aorig_hffp32 hf_fp32 q1_Aorig $ORIG --bs 8
sc q1_A_hfbf16 hf_bf16 q1_A $DEQ --bs 8
sc q1_A_hffp32 hf_fp32 q1_A $DEQ --bs 8
sc q1_B_hfbf16 hf_bf16 q1_B $DEQ --bs 8
sc q1_B_hffp32 hf_fp32 q1_B $DEQ --bs 8
sc q1_B_engine engine q1_B $DEQ --bs 8

# ---- Q2: fixed length via ignore_eos; N seqs = prompts 0..N-1, G = 1 ----
for cfg in 256:64:8 1024:32:1 4096:16:1 16384:8:1; do
  IFS=: read L N BS <<< "$cfg"
  roll q2_${L}_A bf16 $DEQ --n-prompts $N --g 1 --max-new $L --ignore-eos
  roll q2_${L}_B bitexact $INT_ORIG --n-prompts $N --g 1 --max-new $L --ignore-eos
  sc q2_${L}_A_hfbf16 hf_bf16 q2_${L}_A $DEQ --bs $BS
  sc q2_${L}_A_hffp32 hf_fp32 q2_${L}_A $DEQ --bs $BS
  sc q2_${L}_B_hfbf16 hf_bf16 q2_${L}_B $DEQ --bs $BS
  sc q2_${L}_B_hffp32 hf_fp32 q2_${L}_B $DEQ --bs $BS
  sc q2_${L}_B_engine engine q2_${L}_B $DEQ --bs 1
done
echo ALLDONE
