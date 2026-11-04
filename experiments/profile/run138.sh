#!/usr/bin/env bash
# All profiling runs on one H100 of host 138, sequentially. GPU: first argument, else the last card with 0 MiB used.
source $(dirname $0)/env138.sh
G=${1:-$(freegpus | awk '{print $NF}')}
[ -n "$G" ] || { echo "no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$G
echo "GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) $(date +%T)"
step() { local out=$1; shift; [ -f $out ] && { echo "skip $out"; return; }; echo "== $(date +%T) $out"; "$@" || echo "FAIL $out"; }
SC="$PY $X/prof_scorer.py --const $CONST --model $DEQ --data $DATA"
GSM=$P/roll/q1_B.json; L4K=$P/roll/q2_4096_B.json
step $O/sc_gsm_b8_eager.json   $SC --rollout $GSM --bs 8 --reps 3 --prof-batches 4 --bytes --out $O/sc_gsm_b8_eager
step $O/sc_gsm_b8_comp.json    $SC --rollout $GSM --bs 8 --reps 1 --prof-batches 4 --compiled --out $O/sc_gsm_b8_comp
for b in 1 4 8; do
  step $O/sc_4k_b${b}_eager.json $SC --rollout $L4K --bs $b --reps 2 --prof-batches 1 $([ $b = 4 ] && echo --bytes) --out $O/sc_4k_b${b}_eager
done
step $O/sc_4k_b4_comp.json     $SC --rollout $L4K --bs 4 --reps 1 --prof-batches 1 --compiled --out $O/sc_4k_b4_comp
for B in 1 64; do for c in 512 2304; do for md in eager compiled; do
  step $O/dec_B${B}_c${c}_${md}.json $PY $X/prof_decode.py --const $CONST --model $DEQ --B $B --ctx $c --mode $md --out $O/dec_B${B}_c${c}_${md}
done; done; done
for B in 1 64; do
  step $O/vllm_B${B}.json $PY $X/prof_vllm_decode.py --model $INT --data $DATA --B $B --gen 512 --prof-gen 128 --out $O/vllm_B${B}.json
done
echo ALLDONE $(date +%T)
