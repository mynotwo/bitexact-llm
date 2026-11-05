#!/usr/bin/env bash
# README throughput table: 1B and 8B, exact / int8 / bf16 stock / bf16 batch-invariant, 5 interleaved reps.
#   bash readme_bench.sh start   (one GPU, empty-checked)  |  bash readme_bench.sh agg
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf9_20261105/merge
A=/data41/audreymao/a17_vllm
D=/data41/audreymao/perf_20261030
P=/data41/audreymao/perf7_20261105/exactplugin
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
E1=$P/exactmodel; E8=$P/exact8bmodel
I1=$D/det_compare/intmodel; I8=/data41/audreymao/p3_8b/intmodel8b
B1=$D/bf16model; B8=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
case "${1:-start}" in
sync) ssh $HOST "mkdir -p $S/rd"; scp -q $LOCAL/experiments/perf/p9_merge/readme_bench.sh $HOST:$S/rd/; exit ;;
start)
  ssh $HOST "export GPU='${GPU:-3}' REPS='${REPS:-5}' SIZES='${SIZES:-1 8}'; nohup bash $S/rd/readme_bench.sh go > $S/go_readme_bench_gpu\${GPU:-3}.log 2>&1 < /dev/null & echo started"; exit ;;
agg)
  ssh $HOST "python3 - <<'PY'
import glob, json, os, statistics as st
d={}
for f in glob.glob('$S/out/rd/*.log'):
    for l in open(f):
        if l.startswith('RESULT '):
            j=json.loads(l[7:]); k=j['tag'].rsplit('_r',1)[0]; d.setdefault(k,[]).append(j['tok_per_s'])
for k in sorted(d): print('%-10s n=%d med=%8.1f %s' % (k, len(d[k]), st.median(d[k]), [round(x) for x in d[k]]))
PY"; exit ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0
GPU=${GPU:-3}; REPS=${REPS:-5}
case " 0 1 2 3 7 " in *" $GPU "*) ;; *) echo "GPU $GPU not assigned"; exit 2;; esac
O=$S/out/rd; mkdir -p $O; SUM=$S/out/SUMMARY_readme_bench_gpu$GPU.txt
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { while ! free $GPU; do sleep 30; done; sleep 3; }
run() { log=$1; to=$2; shift 2
  wait_gpu; for try in 1 2 3 4; do
    ( cd $S/repo && PYTHONPATH=$S/repo/src:$S/repo/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
    rc=$?; grep -q "Error in memory profiling\|No available memory for the cache\|Not enough memory" $log || break
    echo "$(date +%T) retry $try $(basename $log .log)"; wait_gpu; done
  echo "$(date +%T) $(basename $log .log) rc=$rc $(grep -h '^RESULT' $log | cut -c1-120)"; }
for rep in $(seq 1 $REPS); do for sz in ${SIZES:-1 8}; do for b in 1 32 128; do
  eval "E=\$E$sz I=\$I$sz B=\$B$sz"
  run $O/e${sz}_b${b}_r$rep.log 2400 $PY tests/vllm/bench.py --model $E --batch $b --temp 0 --tag e${sz}_b${b}_r$rep
  run $O/i${sz}_b${b}_r$rep.log 2400 $PY tests/vllm/bench.py --model $I --batch $b --temp 0 --tag i${sz}_b${b}_r$rep
  run $O/f${sz}_b${b}_r$rep.log 2400 $PY tests/vllm/bench.py --model $B --batch $b --temp 0 --tag f${sz}_b${b}_r$rep
  run $O/n${sz}_b${b}_r$rep.log 2400 env VLLM_BATCH_INVARIANT=1 $PY tests/vllm/bench.py --model $B --batch $b --temp 0 --attn TRITON_ATTN --tag n${sz}_b${b}_r$rep
done; done; done
echo "$(date +%T) RDONE" | tee -a $SUM
