#!/usr/bin/env bash
# Full A/B/C GRPO runs: 4 arms x 2 seeds x 200 steps, one process per GPU, queued per GPU.
#   source env.sh; OUT=~/abc_rl/runs bash run_full.sh "1:C_s1" "6:C_s2" "7:B_s1" "2:B_s2" "3:A_s1,A_TIS_s1" "4:A_s2,A_TIS_s2"
# Each argument is GPU:run[,run...]; runs on one GPU execute one after another.
set -u
: "${PY:?}" "${C:?}" "${HF:?}" "${CONST:?}" "${DATA:?}" "${OUT:?}"
mkdir -p "$OUT"
for slot in "$@"; do
  gpu=${slot%%:*}; runs=${slot#*:}
  (
    for r in ${runs//,/ }; do
      arm=${r%_s*}; seed=${r##*_s}
      CUDA_VISIBLE_DEVICES=$gpu $PY $C/train.py --arm "$arm" --seed "$seed" --steps 200 --eval-every 25 \
        --hf "$HF" --const "$CONST" --data "$DATA" --work "$OUT/../work" --out "$OUT/$r" > "$OUT/$r.log" 2>&1
      echo "$r exit $? $(date)" >> "$OUT/queue.log"
    done
  ) &
done
wait
