#!/usr/bin/env bash
# Rollouts, rescoring and replays on one 8xH800 host (GPUs given in G0..G3; each needs ~25 GB free).
#   R=/data00/.../rl_mismatch PY=venv/bin/python BF16=bf16_dir INT=intmodel_dir CONST=int8_const.pt \
#   DATA=gsm_dir G0=0 G1=1 G2=2 G3=3 bash run_141.sh
set -u
C=$(cd "$(dirname "$0")" && pwd); O=$R/out; mkdir -p $O
run() { g=$1; n=$2; shift 2; CUDA_VISIBLE_DEVICES=$g timeout 5400 "$@" --out $O/$n.json > $O/$n.log 2>&1; grep -h RESULT $O/$n.log || echo "FAIL $n"; }
V="$PY $C/rollout_vllm.py --data $DATA"
HF="$PY $C/score_hf.py --model $BF16 --data $DATA"
BX="$PY $C/bitexact_engine.py --const $CONST --tokenizer $INT --data $DATA"
(   run $G0 r0_bf16 $V --engine bf16 --model $BF16
    run $G0 r0b_bf16_repeat $V --engine bf16 --model $BF16
    run $G0 r1_bf16_shuf32 $V --engine bf16 --model $BF16 --order shuffle --max-seqs 32
    run $G0 r2_bf16_seq8 $V --engine bf16 --model $BF16 --max-seqs 8 ) &
(   run $G1 b0_bx $V --engine bitexact --model $INT
    run $G1 b1_bx_shuf32 $V --engine bitexact --model $INT --order shuffle --max-seqs 32 ) &
wait
(   run $G2 s_hf_bf16_bs8 $HF --rollout $O/r0_bf16.json --dtype bf16 --bs 8 --order group
    run $G2 s_hf_bf16_bs1 $HF --rollout $O/r0_bf16.json --dtype bf16 --bs 1 --order group
    run $G2 s_hf_bf16_bs32 $HF --rollout $O/r0_bf16.json --dtype bf16 --bs 32 --order sorted
    run $G2 s_hf_fp32_bs8 $HF --rollout $O/r0_bf16.json --dtype fp32 --bs 8 --order group ) &
(   run $G3 s_bx_h800_bs8 $BX score --rollout $O/b0_bx.json --device cuda --bs 8 --order group
    run $G3 s_bx_h800_bs32 $BX score --rollout $O/b0_bx.json --device cuda --bs 32 --order sorted ) &
(   run $G1 g_bx_h800_bs64 $BX generate --rollout $O/b0_bx.json --device cuda --bs 64 ) &
wait
echo ALLDONE
