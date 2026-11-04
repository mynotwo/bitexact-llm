#!/usr/bin/env bash
# Q2 metrics (per position bucket, cumulative drift) from the raw files in $O.
. /data41/audreymao/abc_measure/env.sh
args=""
for L in 256 1024 4096 16384; do
  args="$args A_hfbf16_$L=$O/q2_${L}_A.json,$O/s_q2_${L}_A_hfbf16.json A_hffp32_$L=$O/q2_${L}_A.json,$O/s_q2_${L}_A_hffp32.json"
  args="$args B_hfbf16_$L=$O/q2_${L}_B.json,$O/s_q2_${L}_B_hfbf16.json B_hffp32_$L=$O/q2_${L}_B.json,$O/s_q2_${L}_B_hffp32.json C_engine_$L=$O/q2_${L}_B.json,$O/s_q2_${L}_B_engine.json"
done
$PY $C/analyze_len.py --out $O/metrics_q2.json --pair $args
