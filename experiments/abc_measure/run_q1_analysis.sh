#!/usr/bin/env bash
# Q1 metrics from the raw files in $O (same analyze.py as rl_mismatch).
. /data41/audreymao/abc_measure/env.sh
$PY $R/analyze.py --tokenizer $ORIG --data $DATA --out $O/metrics_q1.json --mismatch \
  A_orig_hfbf16=$O/q1_Aorig.json,$O/s_q1_Aorig_hfbf16.json A_orig_hffp32=$O/q1_Aorig.json,$O/s_q1_Aorig_hffp32.json \
  A_hfbf16=$O/q1_A.json,$O/s_q1_A_hfbf16.json A_hffp32=$O/q1_A.json,$O/s_q1_A_hffp32.json \
  B_hfbf16=$O/q1_B.json,$O/s_q1_B_hfbf16.json B_hffp32=$O/q1_B.json,$O/s_q1_B_hffp32.json \
  C_engine=$O/q1_B.json,$O/s_q1_B_engine.json 2>&1 | grep -v transformers
