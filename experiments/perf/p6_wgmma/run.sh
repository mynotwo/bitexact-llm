#!/usr/bin/env bash
# p6_wgmma: sm90 wgmma int8 GEMM verification / bench / profile on 141. One GPU per `go` process.
#   bash run.sh sync [BRANCH] ; GPU=0 STAGES="pytest gemm" TREES="base repo" bash run.sh start ; bash run.sh status
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf6_20261105/wgmma
A=/data41/audreymao/a17_vllm
S8=/data41/audreymao/p3_8b
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
case "${1:-}" in
sync)
  BR=${2:-HEAD}; T=${TREE:-repo}
  ssh $HOST "mkdir -p $S/$T"
  git -C "$LOCAL" archive $BR | ssh $HOST "tar -x -C $S/$T"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BR) > $S/$T/SYNCED_COMMIT"
  ssh $HOST "mkdir -p $S/p6" && scp -q "$LOCAL/experiments/perf/p6_wgmma/"*.py $HOST:$S/p6/ && scp -q "$LOCAL/experiments/perf/p6_wgmma/run.sh" $HOST:$S/ ; exit ;;
start)
  st=${STAGES:-bench}; L=$S/go_${st// /_}_gpu${GPU:?set GPU}.log
  ssh $HOST "export STAGES='$st' GPU='$GPU' TREES='${TREES:-}' REPS='${REPS:-3}' O='${O:-}'; R=$S/run_\$\$.sh; cp $S/run.sh \$R; nohup bash \$R go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 3 $S/go_*.log 2>/dev/null | cut -c1-200; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
CONST=$A/data/intllama_const.pt
C8=$S8/int8_const_8b.pt; M8=$S8/intmodel8b
HC=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-bench}; TREES=${TREES:-repo}; MAXWAIT=${MAXWAIT:-86400}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}_gpu$GPU.txt
: "${GPU:?set GPU}"
case " 0 1 2 3 " in *" $GPU "*) ;; *) echo "GPU $GPU not assigned"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, trees: $TREES" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait GPU $GPU"
    sleep 30; w=$((w + 30)); done
  sleep 3; free $GPU || { wait_gpu; return; }
  echo "$(date +%T) $1 gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') uptime=$(uptime | sed 's/.*load average: //')" >> $O/quiet_check.log; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu $(basename $log .log) || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed\|BATCH_INV' $log | tail -n 1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -n 3 | cut -c1-220; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }
EP=$S/p6

# microbench: cutlass / old Triton table / new sm90, per shape and rows
gemm() { D=$O/gemm; mkdir -p $D
  for t in $TREES; do
    run $D/bench_$t.log 3600 $t $PY $EP/bench_gemm.py --json $D/bench_$t.json \
        --old $S/base/src/bitexact/int8/gemm_skinny.py --rows ${ROWS:-64,128,256,384,2048,8192,32768}
    grep -h '^ROW' $D/bench_$t.log | tee -a $SUM
  done; }

pytest() { D=$O/pytest; mkdir -p $D
  for t in $TREES; do run $D/pytest_$t.log 7200 $t env BITEXACT_INT8_CONST=$CONST BITEXACT_INT8_CONST_8B=$C8 $PY -m pytest -q tests/
    tail -n 5 $D/pytest_$t.log | tee -a $SUM; done; }

suite() { D=$O/suite; mkdir -p $D
  for t in $TREES; do
    wait_gpu suite_$t || return 1
    echo "$(date +%T) start suite $t" | tee -a $SUM
    ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm GPUS=$GPU PY=$PY MODEL=$MODEL CONST=$CONST OUT=$D \
        timeout 36000 bash tests/vllm/run_tests.sh $t ) > $D/$t.out 2>&1
    echo "$(date +%T) end suite $t rc=$?" | tee -a $SUM
    grep -h "logit_hash_diff\|token_seqs_diff\|missing\|compare.*exit\|Traceback" $D/$t.out | sed "s|$D/||" | tee -a $SUM
  done; }
suitecmp() { D=$O/suite; R=${CMPA:-base}; N=${CMPB:-repo}
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    hcmp $D/${R}_$j $D/${N}_$j
  done; }

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 1800 $t $PY $EP/tokens.py --model $MODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 1800 $t $PY $EP/tokens.py --model $MODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 1800 $t $PY $EP/tokens.py --model $MODEL --out $D/${t}_lp.json --logprobs
  done; }
tokcmp() { D=$O/tokens; R=${CMPA:-base}; N=${CMPB:-repo}
  for k in mixed t0 lp; do cmp -s $D/${R}_$k.json $D/${N}_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

# end-to-end decode bench: 128 in / 128 out
e2e() { D=$O/e2e; mkdir -p $D
  T=${T2:-0 1}; B=${BS:-1 32 128}
  for rep in $(seq 1 ${REPS:-5}); do for b in $B; do for t in $TREES; do for tp in $T; do
    run $D/${t}_b${b}_t${tp}_r$rep.log 1800 $t $PY $EP/bench.py --model $MODEL --batch $b --temp $tp --tag ${t}_b${b}_t${tp}_r$rep
  done; done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

# 8B end-to-end decode bench
e2e8() { D=$O/e2e8; mkdir -p $D
  for rep in $(seq 1 ${REPS:-5}); do for b in ${BS:-1 32 128}; do for t in $TREES; do
    run $D/${t}_b${b}_r$rep.log 2400 $t $PY $EP/bench.py --model $M8 --batch $b --temp 0 --tag ${t}_b${b}_r$rep
  done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

prefill() { D=$O/prefill; mkdir -p $D
  for rep in $(seq 1 ${REPS:-3}); do for sh in "64 256" "64 512" "64 1024"; do set -- $sh; b=$1; l=$2
    for t in $TREES; do
      run $D/${t}_b${b}_l${l}_r$rep.log 1800 $t $PY $EP/bench_prefill.py --model $MODEL --batch $b --in-len $l --tag ${t}_b${b}_l${l}_r$rep
    done
  done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

prof() { for sh in "128 0" "128 1"; do set -- $sh; b=$1; e=$2; D=$O/prof_b${b}_e$e
  for rep in ${PREPS:-1 2 3}; do for t in $TREES; do
    mkdir -p $D/$t$rep
    run $D/$t$rep/int.log 2400 $t bash -c "cd $EP && $PY prof_vllm.py --model $MODEL --batch $b --in-len 128 --dec-steps 10 $([ $e = 1 ] && echo --eager) --out $D/$t$rep/int"
    grep -H "^DECODE\|^PREFILL" $D/$t$rep/int.log | cut -c1-600 | tee -a $SUM
  done; done; done; }

prof8() { b=${B8:-128}; D=$O/prof8_b$b
  for rep in ${PREPS:-1 2 3}; do for t in $TREES; do
    mkdir -p $D/$t$rep
    run $D/$t$rep/int.log 3000 $t bash -c "cd $EP && $PY prof_vllm.py --model $M8 --batch $b --in-len 128 --dec-steps 6 --out $D/$t$rep/int"
    grep -H "^DECODE" $D/$t$rep/int.log | cut -c1-600 | tee -a $SUM
  done; done; }

s8() { D=$O/s8; mkdir -p $D; export GPU_UTIL=0.6; RS="$PY tests/vllm/run_serve.py --model $M8"
  for t in $TREES; do
  run $D/${t}_ref.log 3000 $t $PY tests/vllm/run_ref.py --const $C8 --out $D/${t}_ref
  run $D/${t}_alone_pc1.log 3000 $t $RS --mode alone --prefix-cache 1 --out $D/${t}_alone_pc1
  run $D/${t}_busy_pc1_s0.log 3000 $t $RS --mode busy --prefix-cache 1 --arrival-seed 0 --out $D/${t}_busy_pc1_s0
  run $D/${t}_busy_pc0_s1.log 3000 $t $RS --mode busy --prefix-cache 0 --arrival-seed 1 --out $D/${t}_busy_pc0_s1
  run $D/${t}_busy_pc1_s2_b64.log 3000 $t $RS --mode busy --prefix-cache 1 --arrival-seed 2 --budget 64 --out $D/${t}_busy_pc1_s2_b64
  run $D/${t}_busy_eager_s4.log 3000 $t env BITEXACT_GRAPHS=0 BITEXACT_COMPILE=0 $RS --mode busy --prefix-cache 1 --arrival-seed 4 --max-seqs 128 --out $D/${t}_busy_eager_s4
  run $D/${t}_busy_vllmgraph_s5.log 3000 $t $RS --mode busy --prefix-cache 1 --arrival-seed 5 --eager 0 --out $D/${t}_busy_vllmgraph_s5
  run $D/${t}_alone_vllmgraph.log 3000 $t $RS --mode alone --prefix-cache 1 --eager 0 --out $D/${t}_alone_vllmgraph
  ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm $PY tests/vllm/compare.py $D/${t}_ref $D/${t}_alone_pc1 $D/${t}_busy_pc1_s0 \
      $D/${t}_busy_pc0_s1 $D/${t}_busy_pc1_s2_b64 $D/${t}_busy_eager_s4 $D/${t}_busy_vllmgraph_s5 $D/${t}_alone_vllmgraph ) > $D/${t}_compare.txt 2>&1
  echo "s8_$t compare_exit $?" | tee -a $SUM; tail -n 4 $D/${t}_compare.txt | tee -a $SUM
  done; unset GPU_UTIL; }

bi8() { D=$O/bi8; mkdir -p $D
  for t in $TREES; do
  run $D/${t}_bi_graph.log 3000 $t $PY experiments/perf/p3_8b/batch_inv.py --model $M8 --out $D/${t}_bi_graph.json --eager 0
  run $D/${t}_bi_eager.log 3000 $t env BITEXACT_GRAPHS=0 $PY experiments/perf/p3_8b/batch_inv.py --model $M8 --out $D/${t}_bi_eager.json --eager 1
  done
  grep -H "BATCH_INV" $D/*.log | tee -a $SUM; }

bi1() { D=$O/bi1; mkdir -p $D
  for t in $TREES; do
  run $D/${t}_bi_graph.log 3000 $t $PY experiments/perf/p3_8b/batch_inv.py --model $MODEL --out $D/${t}_bi_graph.json --eager 0
  run $D/${t}_bi_eager.log 3000 $t env BITEXACT_GRAPHS=0 $PY experiments/perf/p3_8b/batch_inv.py --model $MODEL --out $D/${t}_bi_eager.json --eager 1
  done
  grep -H "BATCH_INV" $D/*.log | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM