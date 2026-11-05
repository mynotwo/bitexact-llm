#!/usr/bin/env bash
# p8_exactfuse: verification / benches / profiles of exact-mode fused element-wise kernels on 141.
# One GPU per `go` process, GPU-empty check before every launch.
#   STAGES="pytest" GPU=0 bash run.sh start ; STAGES="suite suitecmp" GPU=1 bash run.sh start ; bash run.sh status
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf8_20261105/exactfuse
A=/data41/audreymao/a17_vllm
D=/data41/audreymao/perf_20261030
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
EMODEL=/data41/audreymao/perf7_20261105/exactplugin/exactmodel
ECONST=$D/exact_tput/consts/exact.pt
IMODEL=$D/det_compare/intmodel
ICONST=$A/data/intllama_const.pt
BF16=$D/bf16model
BASEC=175ff6c
case "${1:-}" in
sync)
  BR=${2:-HEAD}; T=${TREE:-repo}
  ssh $HOST "mkdir -p $S/$T"
  rsync -a --delete --exclude .git --exclude runs "$LOCAL/" $HOST:$S/$T/
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BR) > $S/$T/SYNCED_COMMIT"
  exit ;;
start)
  st=${STAGES:-pytest}; L=$S/go_${st// /_}_gpu${GPU:?set GPU}.log
  ssh $HOST "export STAGES='$st' GPU='$GPU' TREES='${TREES:-repo}' REPS='${REPS:-5}' O='${O:-}'; R=$S/run_\$\$.sh; cp $S/repo/experiments/perf/p8_exactfuse/run.sh \$R; nohup bash \$R go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 4 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-pytest}; TREES=${TREES:-repo}; MAXWAIT=${MAXWAIT:-86400}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}_gpu$GPU.txt
: "${GPU:?set GPU}"
case " 0 1 2 3 " in *" $GPU "*) ;; *) echo "GPU $GPU not assigned"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, trees: $TREES, commit $(cat $S/repo/SYNCED_COMMIT 2>/dev/null)" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done
  sleep 3; free $GPU || { wait_gpu; return; }
  echo "$(date +%T) ${1:-run} gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') uptime=$(uptime | sed 's/.*load average: //')" >> $O/quiet_check.log; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...
  wait_gpu $(basename $log .log) || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed\|BATCH_INV' $log | tail -n 1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -n 3 | cut -c1-220; }
cmp_tree() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }

pytest() { D=$O/pytest; mkdir -p $D
  for t in $TREES; do
    run $D/pytest_$t.log 10800 $t env BITEXACT_EXACT_CONST=$ECONST BITEXACT_INT8_CONST=$ICONST $PY -m pytest -q tests/
    tail -n 6 $D/pytest_$t.log | tee -a $SUM; done; }

suite() { D=$O/suite; mkdir -p $D
  for t in base repo; do
    wait_gpu suite_$t || return 1
    echo "$(date +%T) start suite $t" | tee -a $SUM
    ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm GPUS=$GPU PY=$PY MODEL=$EMODEL CONST=$ECONST OUT=$D \
        timeout 36000 bash tests/vllm/run_tests.sh $t ) > $D/$t.out 2>&1
    echo "$(date +%T) end suite $t rc=$?" | tee -a $SUM
    grep -h "logit_hash_diff\|token_seqs_diff\|missing\|compare.*exit\|Traceback" $D/$t.out | sed "s|$D/||" | tee -a $SUM
  done; }
suitecmp() { D=$O/suite; R=${CMPA:-base}; N=${CMPB:-repo}
  cmp_tree $D/${R}_ref $D/${N}_ref $(for j in alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 \
    busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5 alone_vllmgraph; do echo $D/${N}_$j; done)
  cmp_tree $D/${R}_long_ref $D/${N}_long_ref $D/${N}_long_alone $D/${N}_long_busy $D/${N}_long_busy_eager
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    hcmp $D/${R}_$j $D/${N}_$j
  done; }

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 2400 $t $PY experiments/perf/p4_merge/tokens.py --model $EMODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 2400 $t $PY experiments/perf/p4_merge/tokens.py --model $EMODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 2400 $t $PY experiments/perf/p4_merge/tokens.py --model $EMODEL --out $D/${t}_lp.json --logprobs
  done; }
tokcmp() { D=$O/tokens; R=${CMPA:-base}; N=${CMPB:-repo}
  for k in mixed t0 lp; do cmp -s $D/${R}_$k.json $D/${N}_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

# end-to-end: 1B exact repo vs int8 plugin / bf16 stock / bf16 BI+TRITON, interleaved, 128 in / 128 out, T=0 and T=1
e2e() { D=$O/e2e; mkdir -p $D
  for rep in $(seq 1 ${REPS:-5}); do for b in ${BATCHES:-1 32 128}; do for cfg in ${CONFIGS:-exact int8 bf16 bi}; do
    case $cfg in
      exact) pre="";      cmd="$PY tests/vllm/bench.py --model $EMODEL";;
      int8)  pre="";      cmd="$PY tests/vllm/bench.py --model $IMODEL";;
      bf16)  pre="";      cmd="$PY tests/vllm/bench.py --model $BF16";;
      bi)    pre="env VLLM_BATCH_INVARIANT=1"; cmd="$PY tests/vllm/bench.py --model $BF16 --attn TRITON_ATTN";;
    esac
    for t in ${TEMPS:-0 1}; do
      [ $cfg = int8 ] && [ $t = 1 ] && continue
      run $D/${cfg}_b${b}_t${t}_r$rep.log 1800 repo $pre $cmd --batch $b --temp $t --tag ${cfg}_b${b}_t${t}_r$rep
    done
  done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

prof() { for b in ${PB:-128 32 1}; do for tp in ${PT:-0}; do for t in $TREES; do D=$O/prof_b${b}_t$tp
    mkdir -p $D
    pv=experiments/perf/p8_exactfuse/prof_vllm.py
    [ -f $S/$t/$pv ] || pv=experiments/perf/p7_exactplugin/prof_vllm.py
    run $D/$t.log 2400 $t bash -c "cd $(dirname $pv) && $PY prof_vllm.py --model $EMODEL --batch $b --temp $tp --tag $t --out $D/$t"
    grep -h "^{" $D/$t.log | cut -c1-1600 | tee -a $SUM
  done; done; done; }

# microbench: kernel time of the fused stages and of the fused linear, per row count
micro() { D=$O/micro; mkdir -p $D
  run $D/ew.log 2400 repo $PY experiments/perf/p8_exactfuse/bench_ew.py $ECONST 1,8,32,128
  grep -h "^M=" $D/ew.log | tee -a $SUM
  run $D/gemm.log 2400 repo $PY experiments/perf/p8_exactfuse/bench_gemm.py $ECONST --rows 1,8,32,64,128
  grep -h " us " $D/gemm.log | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM