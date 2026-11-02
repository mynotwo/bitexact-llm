#!/usr/bin/env bash
# rerun.sh GPU: after verify.sh ALLDONE, rerun suite jobs whose run failed for environment reasons, then compare
set -u
V=/data41/audreymao/b128_20261031/attn/verify; A=/data41/audreymao/a17_vllm; PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel; O=$V/out/suite; G=$1
REF=/data41/audreymao/speedup_20261030/combo/out/suite
until grep -q ALLDONE $V/driver.log; do sleep 30; done
. $A/env.sh; export VLLM_ENABLE_V1_MULTIPROCESSING=0
for tj in "$@"; do [ "$tj" = "$G" ] && continue; t=${tj%%:*}; j=${tj#*:}
  case $j in alone_pc0) X="--mode alone --prefix-cache 0";; *) echo "unknown $j"; continue;; esac
  until [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")" = 0 ]; do sleep 10; done
  echo "$(date +%T) rerun ${t}_$j"
  ( cd $V/$t && PYTHONPATH=$V/$t/src:$V/$t/tests/vllm CUDA_VISIBLE_DEVICES=$G timeout 2400 $PY tests/vllm/run_serve.py --model $MODEL $X --out $O/${t}_$j ) > $O/${t}_$j.rerun.log 2>&1
  echo "rc=$?"; (cd $V/repo && $PY tests/vllm/compare.py $O/${t}_ref $O/${t}_$j)
  cmp -s $O/repo_$j.hash $O/base_$j.hash && echo "$j repo vs base IDENT" || echo "$j repo vs base DIFF"
  cmp -s $O/${t}_$j.hash $REF/repo_$j.hash && echo "$j ${t} vs combo IDENT" || echo "$j ${t} vs combo DIFF"
done
echo RERUNDONE
