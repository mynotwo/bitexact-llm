#!/usr/bin/env bash
# p9_merge: combined verification of the four performance lanes merged into main, on 141.
#   bash run.sh sync base 175ff6c ; bash run.sh sync repo main
#   STAGES="pytest suite suitecmp tokens tokcmp check_plug" TREES="base repo" GPU=0 bash run.sh start
#
# Every stage compares two trees of the same lane directory: `base` is the main commit the lanes forked from, `repo`
# is main with the lanes merged. Nothing here changes the trees; the numbers are the gate for the merge.
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf9_20261105/merge
A=/data41/audreymao/a17_vllm
S8=/data41/audreymao/p3_8b
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
case "${1:-}" in
sync)
  T=${2:?tree base|repo}; R=${3:?rev}
  ssh $HOST "mkdir -p $S && rm -rf $S/$T && mkdir -p $S/$T"
  git -C "$LOCAL" archive $R | ssh $HOST "tar -x -C $S/$T"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $R) > $S/$T/SYNCED_COMMIT"; exit ;;
start)
  st=${STAGES:-pytest}; L=$S/go_${st// /_}_gpu${GPU:?set GPU}.log
  ssh $HOST "export STAGES='$st' GPU='$GPU' TREES='${TREES:-}' REPS='${REPS:-3}' O='${O:-}'; R=$S/run_\$\$_\${st// /_}.sh; cp $S/p9/run.sh \$R; nohup bash \$R go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 4 $S/go_*.log 2>/dev/null; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
CONST=$A/data/intllama_const.pt
C8=$S8/int8_const_8b.pt; M8=$S8/intmodel8b
HF=/data41/audreymao/abc_rl/hf DATA=/data41/audreymao/abc_rl/data
ECONST=/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt
HF1=/data41/audreymao/perf_20261030/bf16model
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-pytest}; TREES=${TREES:-base repo}; MAXWAIT=${MAXWAIT:-86400}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}_gpu$GPU.txt
: "${GPU:?set GPU}"
case " 0 1 2 3 7 " in *" $GPU "*) ;; *) echo "GPU $GPU not assigned"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, trees: $TREES, base $(cat $S/base/SYNCED_COMMIT 2>/dev/null) repo $(cat $S/repo/SYNCED_COMMIT 2>/dev/null)" | tee -a $SUM

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
  for try in 1 2 3 4; do      # vLLM's memory profiling asserts when a neighbour job frees GPU memory meanwhile: retry
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; grep -q "Error in memory profiling\|No available memory for the cache\|Not enough memory" $log || break; echo "$(date +%T) retry $try"; wait_gpu retry; done
  echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed\|BATCH_INV\|^STACK\|^PROF' $log | tail -n 1 | cut -c1-260)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -n 3 | cut -c1-220; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }
EP=$S/p9

pytest() { D=$O/pytest; mkdir -p $D
  for t in $TREES; do run $D/pytest_$t.log 7200 $t env BITEXACT_INT8_CONST=$CONST BITEXACT_INT8_CONST_8B=$C8 BITEXACT_EXACT_CONST=$ECONST BITEXACT_MODEL=$HF1 BITEXACT_VLLM_MODEL=$MODEL $PY -m pytest -q tests/
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

suitecmp() { D=$O/suite; R=base; N=repo
  for j in alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager ref; do
    hcmp $D/${R}_$j $D/${N}_$j
  done
}

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 1800 $t $PY $S/repo/experiments/perf/p4_merge/tokens.py --model $MODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 1800 $t $PY $S/repo/experiments/perf/p4_merge/tokens.py --model $MODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 1800 $t $PY $S/repo/experiments/perf/p4_merge/tokens.py --model $MODEL --out $D/${t}_lp.json --logprobs
  done; }

tokcmp() { D=$O/tokens; for j in mixed t0 lp; do
    if cmp -s $D/base_$j.json $D/repo_$j.json; then echo "tokens $j IDENTICAL"; else echo "tokens $j DIFF"; fi | tee -a $SUM
  done; }

check_plug() { D=$O/cp; mkdir -p $D $S/cache/triton $S/cache/inductor $S/cache/vllm $S/tmp $S/work_cp
  wait_gpu check_plug || return 1
  ( cd $S/repo/experiments/abc_rl && env PYTHONPATH=$S/repo/src TRITON_CACHE_DIR=$S/cache/triton \
      TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor VLLM_CACHE_ROOT=$S/cache/vllm TMPDIR=$S/tmp \
      XDG_CACHE_HOME=$S/cache FLASHINFER_WORKSPACE_BASE=$S/cache VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY check_plug.py --hf $HF --const $CONST --data $DATA \
      --work $S/work_cp --out $D/res ) > $D/check_plug.log 2>&1
  echo "check_plug rc=$?" | tee -a $SUM
  $PY -c "import json;d=json.load(open('$D/res'));[print({k:v for k,v in r.items() if 'differ' in k or k in ('update','n_tok')}) for r in d]" | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) MERGEDONE" | tee -a $SUM