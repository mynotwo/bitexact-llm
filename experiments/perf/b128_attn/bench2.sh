#!/usr/bin/env bash
# bench2.sh GPU: clean same-GPU timing (verify.sh GPU 7 was shared with the elt lane). Logs foreign GPU procs per run.
set -u
V=/data41/audreymao/b128_20261031/attn/verify; A=/data41/audreymao/a17_vllm; PY=$A/venv/bin/python
CONST=$A/data/intllama_const.pt; MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
O=$V/out2; mkdir -p $O/prof; G=$1
case " 2 3 4 5 6 " in *" $G "*) exit 2;; esac
. $A/env.sh; export VLLM_ENABLE_V1_MULTIPROCESSING=0
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
free() { [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")" = 0 ] &&
         [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
run() { log=$1; tree=$2; shift 2; for i in $(seq 1440); do free && break; sleep 10; done
  ( cd $V/$tree && PYTHONPATH=$V/$tree/src:$V/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$G "$@" ) > $log 2>&1 & pid=$!
  mx=1; while kill -0 $pid 2>/dev/null; do n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $UUID); [ $n -gt $mx ] && mx=$n; sleep 2; done
  wait $pid; echo "$(date +%T) $(basename $log) rc=$? max_gpu_procs=$mx"; }
for rep in 1 2 3; do for b in 1 32 128; do for t in base repo; do
  run $O/bench_${t}_b${b}_t0_r$rep.log $t timeout 1800 $PY tests/vllm/bench.py --model $MODEL --batch $b --temp 0 --tag ${t}_b${b}_t0
done; done; done
for rep in 1 2; do for b in 1 8; do for t in base repo; do
  run $O/lc_${t}_s16k_b${b}_r$rep.log $t timeout 1200 $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S 16384 --B $b --prof
  grep -o "decode_ms_per_step[^,]*" $O/lc_${t}_s16k_b${b}_r$rep.log
done; done; done
for b in 128 32; do for t in base repo; do
  run $O/prof/prof_${t}_b$b.log $t timeout 2400 $PY experiments/perf/profile/prof_vllm.py --model $MODEL --batch $b --tag ${t}_b$b --out $O/prof/prof_${t}_b$b
done; done
$PY $V/repo/experiments/perf/profile/buckets.py --report $O/prof > $O/prof_summary.md 2>&1; echo "buckets rc=$?"
echo "$(date +%T) ALLDONE"
