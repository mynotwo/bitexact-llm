#!/usr/bin/env bash
# Second GPU type (H100) and x86 CPU: BitExact rescoring and replay generation of the H800 plugin rollouts.
#   R=~/rl_mismatch PY=python CONST=int8_const.pt TOK=tokenizer_dir DATA=gsm_dir GPU=7 [VPY=vllm_python INT=intmodel] bash run_138.sh
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
# Stock vLLM 0.32.0 and the plugin on H100 (VPY: a venv with vllm==0.32.0 and `pip install --no-deps -e .`
# of this repo; INT: make_model_dir output on this host), then the scorer again with that venv's numpy 2.3.5
# (same numpy as the H800 sampler, so the float64 log-softmax step is the same code on both sides).
if [ -n "${VPY:-}" ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=$R/cache VLLM_CACHE_ROOT=$R/cache/vllm PYTHONPATH=
  V="$VPY $C/rollout_vllm.py --data $DATA"
  BX2="$VPY $C/bitexact_engine.py --const $CONST --tokenizer $TOK --data $DATA"
  export CUDA_VISIBLE_DEVICES=$GPU
  run r3_bf16_h100 $V --engine bf16 --model $TOK
  run b2_bx_h100 $V --engine bitexact --model $INT
  run s_bx_h100_np2 $BX2 score --rollout $O/b0_bx.json --device cuda --bs 16 --order sorted
  CUDA_VISIBLE_DEVICES= run s_bx_cpu_np2 $BX2 score --rollout $O/b0_bx.json --device cpu --bs 4 --seeds-from $O/cpu_seeds.txt --threads 48
fi
echo ALLDONE
