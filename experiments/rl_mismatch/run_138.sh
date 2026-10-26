#!/usr/bin/env bash
# Second GPU type (H100) and x86 CPU: BitExact rescoring and replay generation of the H800 plugin rollouts.
#   R=~/rl_mismatch PY=python CONST=int8_const.pt TOK=tokenizer_dir DATA=gsm_dir GPU=7 bash run_138.sh
# Needs $R/out/b0_bx.json (H800 plugin rollout) and $R/out/cpu_seeds.txt (CPU subset).
set -u
C=$(cd "$(dirname "$0")" && pwd); O=$R/out
BX="$PY $C/bitexact_engine.py --const $CONST --tokenizer $TOK --data $DATA"
run() { n=$1; shift; timeout 20000 "$@" --out $O/$n.json > $O/$n.log 2>&1; grep -h RESULT $O/$n.log || echo "FAIL $n"; }
( export CUDA_VISIBLE_DEVICES=$GPU
  run s_bx_h100_bs16 $BX score --rollout $O/b0_bx.json --device cuda --bs 16 --order sorted
  run g_bx_h100_bs16 $BX generate --rollout $O/b0_bx.json --device cuda --bs 16 ) &
( export CUDA_VISIBLE_DEVICES=
  run s_bx_cpu $BX score --rollout $O/b0_bx.json --device cpu --bs 4 --seeds-from $O/cpu_seeds.txt --threads 48
  run g_bx_cpu $BX generate --rollout $O/b0_bx.json --device cpu --bs 4 --seeds-from $O/cpu_seeds.txt --threads 48 ) &
wait
echo ALLDONE
