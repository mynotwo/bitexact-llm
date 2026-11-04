#!/usr/bin/env bash
# Arm D: vLLM bf16 with VLLM_BATCH_INVARIANT=1 + TRITON_ATTN as sampler AND scorer (prompt_logprobs). ONE GPU.
# Resumable. Sampler: max_num_seqs 256, canonical order. Scorer: max_num_seqs 128, shuffled order.
. /data41/audreymao/abc_measure/env.sh
export VLLM_BATCH_INVARIANT=1
cd $W; mkdir -p $O
step() { out=$1; shift; if [ -f "$out" ]; then echo "SKIP $out"; return; fi; echo "RUN $out $(date +%T)"; "$@" || echo "FAIL $out"; }
roll() { name=$1; shift; step $O/$name.json $PY $C/rollout.py --engine bf16 --attn TRITON_ATTN --model $DEQ --data $DATA --out $O/$name.json "$@"; }
sc() { name=$1; roll=$2; shift 2; step $O/s_$name.json $PY $C/score_vllm.py --attn TRITON_ATTN --rollout $O/$roll.json --model $DEQ --data $DATA --max-seqs 128 --out $O/s_$name.json "$@"; }
roll q1_D
sc q1_D_vllm q1_D
for cfg in 256:64 1024:32 4096:16 16384:8; do
  IFS=: read L N <<< "$cfg"
  roll q2_${L}_D --n-prompts $N --g 1 --max-new $L --ignore-eos
  sc q2_${L}_D_vllm q2_${L}_D
done
# speed
step $O/speed_samp_D.json $PY $C/speed_sampler.py --engine bf16 --attn TRITON_ATTN --model $DEQ --data $DATA --out $O/speed_samp_D.json
step $O/speed_sc_gsm_D8.json $PY $C/score_vllm.py --attn TRITON_ATTN --rollout $O/q1_D.json --model $DEQ --data $DATA --max-seqs 8 --reps 4 --no-save --out $O/speed_sc_gsm_D8.json
step $O/speed_sc_gsm_D64.json $PY $C/score_vllm.py --attn TRITON_ATTN --rollout $O/q1_D.json --model $DEQ --data $DATA --max-seqs 64 --reps 4 --no-save --out $O/speed_sc_gsm_D64.json
step $O/speed_sc_4k_D4.json $PY $C/score_vllm.py --attn TRITON_ATTN --rollout $O/q2_4096_D.json --model $DEQ --data $DATA --max-seqs 4 --reps 4 --no-save --out $O/speed_sc_4k_D4.json
echo ALLDONE
