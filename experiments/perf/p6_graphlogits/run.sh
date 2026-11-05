#!/usr/bin/env bash
# p6_graphlogits: profile / bench / verification for the graphlogits lane on 141.
#   bash run.sh sync [BRANCH] [TREE] ; STAGES="prof" TREES="base repo" GPU=0 bash run.sh start ; bash run.sh status
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf6_20261105/graphlogits
A=/data41/audreymao/a17_vllm
S8=/data41/audreymao/p3_8b
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
case "${1:-}" in
sync)
  BR=${2:-HEAD}; T=${3:-repo}
  ssh $HOST "mkdir -p $S && rm -rf $S/$T && mkdir -p $S/$T"
  git -C "$LOCAL" archive $BR | ssh $HOST "tar -x -C $S/$T"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BR) > $S/$T/SYNCED_COMMIT"
  ssh $HOST "mkdir -p $S/p6"; scp -q "$LOCAL/experiments/perf/p6_graphlogits/"{run.sh,stacks.py} $HOST:$S/p6/ 2>/dev/null
  scp -q "$LOCAL/experiments/perf/p5_prefill/"{prof_vllm.py,buckets.py} $HOST:$S/p6/
  scp -q "$LOCAL/experiments/perf/p4_merge/"{tokens.py,bench_attn.py} $HOST:$S/ ; exit ;;
start)
  st=${STAGES:-prof}; L=$S/go_${st// /_}_gpu${GPU:?set GPU}.log
  ssh $HOST "export BITEXACT_GL_CHECK='${GLCHK:-0}' STAGES='$st' GPU='$GPU' TREES='${TREES:-}' REPS='${REPS:-3}' O='${O:-}' PROFARGS='${PROFARGS:-}'; R=$S/run_\$\$_${st// /_}.sh; cp $S/p6/run.sh \$R; nohup bash \$R go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 4 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
CONST=$A/data/intllama_const.pt
C8=$S8/int8_const_8b.pt; M8=$S8/intmodel8b
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-prof}; TREES=${TREES:-repo}; MAXWAIT=${MAXWAIT:-86400}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}_gpu$GPU.txt
: "${GPU:?set GPU}"
case " 0 1 2 3 7 " in *" $GPU "*) ;; *) echo "GPU $GPU not assigned"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, trees: $TREES" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done
  sleep 3; free $GPU || { wait_gpu "$@"; return; }
  echo "$(date +%T) $1 gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') uptime=$(uptime | sed 's/.*load average: //')" >> $O/quiet_check.log; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu $(basename $log .log) || return 1
  echo "$(date +%T) start $(basename $log .log)"
  for try in 1 2 3 4; do      # vLLM's memory profiling asserts when a neighbour lane frees GPU memory meanwhile: retry
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; grep -q "Error in memory profiling" $log || break; echo "$(date +%T) retry $try"; wait_gpu retry; done; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed\|BATCH_INV\|^STACK\|^PROF' $log | tail -n 1 | cut -c1-260)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -n 3 | cut -c1-220; }
cmp_tree() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }
EP=$S/p6

case "$(basename "$0")" in run*.sh) EP=$S/p6 ;; esac

prof() { D=$O/prof; mkdir -p $D
  for cfg in "128 0" "128 1" "32 0" "1 0"; do set -- $cfg; b=$1; t=$2
    for rep in 1 2 3; do for tr in $TREES; do
      mkdir -p $D/${tr}_b${b}_t${t}_r$rep
      run $D/${tr}_b${b}_t${t}_r$rep/int.log 2400 $tr bash -c "cd $EP && $PY prof_vllm.py --model $MODEL --batch $b --temp $t --tag ${tr}_b${b}_t${t} --out $D/${tr}_b${b}_t${t}_r$rep/int"
      grep -H "^TIMING\|^RESULT" $D/${tr}_b${b}_t${t}_r$rep/int.log | cut -c1-400 | tee -a $SUM
    done; done
  done
  bash $EP/agg.sh $D | tee -a $SUM; }

stacks() { D=$O/stacks; mkdir -p $D
  for cfg in "128 0" "128 1" "32 0"; do set -- $cfg; b=$1; t=$2
    for tr in $TREES; do
      run $D/${tr}_b${b}_t${t}.log 2400 $tr bash -c "cd $EP && $PY stacks.py --model $MODEL --batch $b --temp $t --out $D/${tr}_b${b}_t${t}"
      grep -H "^STACK" $D/${tr}_b${b}_t${t}.log | cut -c1-500 | tee -a $SUM
    done
  done; }

bench() { D=$O/bench; mkdir -p $D
  for rep in $(seq 1 ${REPS:-5}); do
    for cfg in "1 0" "32 0" "128 0" "1 1" "32 1" "128 1"; do set -- $cfg; b=$1; t=$2
      for tr in $TREES; do
        run $D/${tr}_b${b}_t${t}_r$rep.log 1800 $tr $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $t --tag ${tr}_b${b}_t${t}_r$rep
      done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

bench8() { D=$O/bench8; mkdir -p $D; export GPU_UTIL=0.5
  for rep in $(seq 1 ${REPS:-5}); do for b in 1 32 128; do for tr in $TREES; do
    run $D/${tr}_b${b}_r$rep.log 2400 $tr env GPU_UTIL=0.5 $PY tests/vllm/bench.py --model $M8 --batch $b --temp 0 --tag ${tr}_b${b}_r$rep
  done; done; done
  unset GPU_UTIL
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

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
  cmp_tree $D/${R}_ref $D/${N}_ref $(for j in alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 \
    busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5 alone_vllmgraph; do echo $D/${N}_$j; done)
  cmp_tree $D/${R}_long_ref $D/${N}_long_ref $D/${N}_long_alone $D/${N}_long_busy $D/${N}_long_busy_eager
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    hcmp $D/${R}_$j $D/${N}_$j
  done; }

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_lp.json --logprobs
  done; }
tokcmp() { D=$O/tokens; R=${CMPA:-base}; N=${CMPB:-repo}
  for k in mixed t0 lp; do cmp -s $D/${R}_$k.json $D/${N}_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

spec() { D=$O/spec; mkdir -p $D; SC="$PY tests/vllm/spec_check.py --model $MODEL --eager 0"
  for k in 0 4; do run $D/suite_k$k.log 2400 repo $SC --work suite --k $k --out $D/suite_k$k; done
  for wk in rep bench; do for B in 1 32; do for T in 0 0.7; do for k in 0 4; do
    run $D/${wk}_B${B}_T${T}_k$k.log 2400 repo $SC --work $wk --batch $B --temp $T --k $k --out $D/${wk}_B${B}_T${T}_k$k
  done; done; done; done
  for f in $D/*_k0.json; do g=${f%_k0.json}_k4.json
    cmp -s $f $g && echo "spec $(basename ${f%_k0.json}) IDENTICAL" || echo "spec $(basename ${f%_k0.json}) DIFF"; done | tee -a $SUM; }

s8() { D=$O/s8; mkdir -p $D; export GPU_UTIL=0.6; RS="$PY tests/vllm/run_serve.py --model $M8"
  run $D/ref.log 3000 repo $PY tests/vllm/run_ref.py --const $C8 --out $D/ref
  run $D/alone_pc1.log 3000 repo $RS --mode alone --prefix-cache 1 --out $D/alone_pc1
  run $D/busy_pc1_s0.log 3000 repo $RS --mode busy --prefix-cache 1 --arrival-seed 0 --out $D/busy_pc1_s0
  run $D/busy_pc0_s1.log 3000 repo $RS --mode busy --prefix-cache 0 --arrival-seed 1 --out $D/busy_pc0_s1
  run $D/busy_pc1_s2_b64.log 3000 repo $RS --mode busy --prefix-cache 1 --arrival-seed 2 --budget 64 --out $D/busy_pc1_s2_b64
  run $D/busy_eager_s4.log 3000 repo env BITEXACT_GRAPHS=0 BITEXACT_COMPILE=0 $RS --mode busy --prefix-cache 1 --arrival-seed 4 --max-seqs 128 --out $D/busy_eager_s4
  run $D/busy_vllmgraph_s5.log 3000 repo $RS --mode busy --prefix-cache 1 --arrival-seed 5 --eager 0 --out $D/busy_vllmgraph_s5
  run $D/alone_vllmgraph.log 3000 repo $RS --mode alone --prefix-cache 1 --eager 0 --out $D/alone_vllmgraph
  ( cd $S/repo && PYTHONPATH=$S/repo/src:$S/repo/tests/vllm $PY tests/vllm/compare.py $D/ref $D/alone_pc1 $D/busy_pc1_s0 $D/busy_pc0_s1 \
      $D/busy_pc1_s2_b64 $D/busy_eager_s4 $D/busy_vllmgraph_s5 $D/alone_vllmgraph ) > $D/compare.txt 2>&1; echo "s8 compare_exit $?" | tee -a $SUM
  tail -n 4 $D/compare.txt | tee -a $SUM
  unset GPU_UTIL; }
bi8() { D=$O/bi8; mkdir -p $D
  run $D/bi_graph.log 3000 repo $PY experiments/perf/p3_8b/batch_inv.py --model $M8 --out $D/bi_graph.json --eager 0
  run $D/bi_eager.log 3000 repo env BITEXACT_GRAPHS=0 $PY experiments/perf/p3_8b/batch_inv.py --model $M8 --out $D/bi_eager.json --eager 1
  grep -H "BATCH_INV" $D/*.log | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM