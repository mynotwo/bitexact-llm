#!/usr/bin/env bash
# verify.sh GPU : independent verification of perf/attn 5601138 (V/repo) vs 29e5cb5 (V/base), all on ONE GPU, sequential.
set -u
V=/data41/audreymao/b128_20261031/attn/verify; A=/data41/audreymao/a17_vllm; PY=$A/venv/bin/python
CONST=$A/data/intllama_const.pt; MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
REF=/data41/audreymao/speedup_20261030/combo/out/suite   # repo_* there = d87b9ce (src/tests identical to 29e5cb5)
O=$V/out; mkdir -p $O; G=$1
case " 2 3 4 5 6 " in *" $G "*) echo "GPU $G not allowed"; exit 2;; esac
. $A/env.sh; export VLLM_ENABLE_V1_MULTIPROCESSING=0
free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
waitg() { for i in $(seq 1440); do free && return 0; sleep 10; done; echo "GPU $G never free"; exit 9; }
run() { log=$1; tree=$2; shift 2; waitg; echo "$(date +%T) start $(basename $log)"
  ( cd $V/$tree && PYTHONPATH=$V/$tree/src:$V/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$G "$@" ) > $log 2>&1
  echo "$(date +%T) end $(basename $log) rc=$?"; }
echo "GPU $G repo $(cat $V/repo/SYNCED_COMMIT) base $(cat $V/base/SYNCED_COMMIT)"
# 1) GPU pytest, repo
run $O/pytest_repo.log repo timeout 2400 env BITEXACT_INT8_CONST=$CONST $PY -m pytest -q -m gpu tests
tail -3 $O/pytest_repo.log
# 2) suite base then repo
for t in base repo; do waitg; echo "$(date +%T) start suite $t"
  ( cd $V/$t && PYTHONPATH=$V/$t/src:$V/$t/tests/vllm GPUS=$G PY=$PY MODEL=$MODEL CONST=$CONST OUT=$O/suite \
    timeout 36000 bash tests/vllm/run_tests.sh $t ) > $O/suite_$t.out 2>&1
  echo "$(date +%T) end suite $t rc=$?"; grep -h "logit_hash_diff\|compare.*exit\|Traceback" $O/suite_$t.out | sed "s|$O/suite/||"
done
for f in $O/suite/repo_*.hash; do n=$(basename $f); n=${n#repo_}
  cmp -s $f $O/suite/base_$n && a=IDENT || a=DIFF; cmp -s $f $REF/repo_$n && b=IDENT || b=DIFF
  echo "$n vs_base29e5cb5=$a vs_combo_d87b9ce=$b"; done | tee $O/hash_cmp.txt
# 3) bench T=0 (and T=1) B=1/32/128, base/repo alternating, 2 reps
for rep in 1 2; do for b in 1 32 128; do for tm in 0 1; do for t in base repo; do
  run $O/bench_${t}_b${b}_t${tm}_r$rep.log $t timeout 1800 $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $tm --tag ${t}_b${b}_t${tm}
  grep -h "^RESULT" $O/bench_${t}_b${b}_t${tm}_r$rep.log | tail -1 | cut -c1-220
done; done; done; done
# 4) longctx S=16k B=1/8
for b in 1 8; do for t in base repo; do
  run $O/lc_${t}_s16k_b$b.log $t timeout 1200 $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S 16384 --B $b --prof
  grep -o "decode_ms_per_step[^,]*" $O/lc_${t}_s16k_b$b.log
done; done
# 5) profile B=128 / 32
mkdir -p $O/prof
for b in 128 32; do for t in base repo; do
  run $O/prof/prof_${t}_b$b.log $t timeout 2400 $PY experiments/perf/profile/prof_vllm.py --model $MODEL --batch $b --tag ${t}_b$b --out $O/prof/prof_${t}_b$b
done; done
$PY $V/repo/experiments/perf/profile/buckets.py --report $O/prof > $O/prof_summary.md 2>&1; echo "buckets rc=$?"
echo "$(date +%T) ALLDONE"
