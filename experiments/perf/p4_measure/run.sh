#!/usr/bin/env bash
# p4_measure: reference numbers on main 6d42e87, GPU 7 only, one run at a time, GPU-empty check + host load per launch.
#   bash run.sh sync ; bash run.sh start ; bash run.sh status      (STAGES="prof bench8 spec")
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/p4_measure
A=/data41/audreymao/a17_vllm
S8=/data41/audreymao/p3_8b
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
PROF_SRC=/Users/bytedance/Documents/projects/bitexact-llm/experiments/perf/profile
case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo && mkdir -p $S/repo"
  git -C "$LOCAL" archive 6d42e87 | ssh $HOST "tar -x -C $S/repo"
  ssh $HOST "mkdir -p $S/repo/experiments/perf/profile"
  scp -q $PROF_SRC/prof_vllm.py $PROF_SRC/buckets.py $HOST:$S/repo/experiments/perf/profile/
  scp -q "$LOCAL/experiments/perf/p3_merge/bench_attn.py" "$LOCAL/experiments/perf/p4_measure/run.sh" $HOST:$S/ ; exit ;;
start)
  ssh $HOST "STAGES='${STAGES:-prof bench8 spec}' nohup bash $S/run.sh go > $S/go.log 2>&1 < /dev/null & echo started"; exit ;;
status) ssh $HOST "tail -n 5 $S/go.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac
. $A/env.sh
PY=$A/venv/bin/python
HF8=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
M8=$S8/intmodel8b
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
export VLLM_ENABLE_V1_MULTIPROCESSING=0
GPU=7; O=$S/out; mkdir -p $O; SUM=$O/summary.txt
free() { [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 10 ] && \
  [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { while ! free; do sleep 30; done; sleep 3; free || { wait_gpu; return; }
  echo "$(date +%T) $1 gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') allgpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',') uptime=$(uptime | sed 's/.*load average: //')" >> $O/quiet_check.log; }
run() { log=$1; to=$2; shift 2; wait_gpu $(basename $log .log)
  ( cd $S/repo && PYTHONPATH=$S/repo/src:$S/repo/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT' $log | tail -n 1 | cut -c1-200)" >> $O/progress.log; }
prof() { for b in 128 32; do for T in 0 1; do for rep in 1 2 3; do D=$O/prof8_b${b}_T$T/r$rep; mkdir -p $D
  run $D/p.log 3000 bash -c "cd experiments/perf/profile && $PY prof_vllm.py --model $M8 --batch $b --temp $T --tag p8_b${b}_T${T} --out $D/int"
  done; done; done; }
bench8() { D=$O/bench8; mkdir -p $D
  for rep in 1 2 3 4 5; do for b in 1 32 128; do
    for T in 0 1; do run $D/plugin_b${b}_T${T}_r$rep.log 2400 $PY tests/vllm/bench.py --model $M8 --batch $b --temp $T --tag plugin8_b${b}_T${T}_r$rep; done
    run $D/stock_b${b}_r$rep.log 2400 $PY $S/bench_attn.py --model $HF8 --batch $b
    run $D/bi_b${b}_r$rep.log 2400 env VLLM_BATCH_INVARIANT=1 $PY $S/bench_attn.py --model $HF8 --batch $b --attn TRITON_ATTN
  done; done; }
spec() { D=$O/spec; mkdir -p $D
  for rep in 1 2 3; do for b in 32 128; do for wk in rep bench; do for T in 0 0.7; do
    ks="0 4"; [ $b = 32 ] && ks="0 4 2"
    for k in $ks; do n=${wk}_B${b}_T${T}_k${k}_r$rep
      run $D/$n.log 2400 $PY tests/vllm/spec_check.py --model $MODEL --eager 0 --batch $b --work $wk --temp $T --k $k --out $D/$n
  done; done; done; done; done
  for f in $D/*_k0_r1.json; do for k in 4 2; do g=${f/_k0_/_k${k}_}; [ -f $g ] && { cmp -s $f $g && echo "spec $(basename ${f/_k0_r1.json/}) k=$k IDENTICAL" || echo "spec $(basename ${f/_k0_r1.json/}) k=$k DIFF"; }; done; done | tee -a $SUM; }
for st in $STAGES; do $st; done
echo "$(date +%T) done" >> $O/progress.log
