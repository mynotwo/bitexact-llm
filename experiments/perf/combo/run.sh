#!/usr/bin/env bash
# perf/combo (fuse-elementwise + sampler-sync) against baseline 48a60f7, plus vLLM bf16 batch-invariant reference,
# all on one GPU, one run at a time.
#
# On the Mac:
#   bash experiments/perf/combo/run.sh sync                       # git archive perf/combo -> $S/repo, 48a60f7 -> $S/base
#   STAGES="suite bench bi" GPU=1 bash experiments/perf/combo/run.sh start
# On 141: STAGES="..." GPU=n bash $S/repo/experiments/perf/combo/run.sh go
#
# STAGES:
#   suite  tests/vllm/run_tests.sh (14 jobs) for both trees; each combo run compared with the baseline run
#   bench  tests/vllm/bench.py B=1/32/128, T=0, both trees (alternating per cell)
#   bi     vLLM bf16, VLLM_BATCH_INVARIANT=1, attention_backend TRITON_ATTN (det_compare/bench_attn.py), B=1/32/128
# Outputs: $O/<stage>/..., $O/SUMMARY_<stages>.txt
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/speedup_20261030/combo
A=/data41/audreymao/a17_vllm
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
BRANCH=${BRANCH:-perf/combo}; BASEC=48a60f7
case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo $S/base && mkdir -p $S/repo $S/base"
  git -C "$LOCAL" archive $BRANCH | ssh $HOST "tar -x -C $S/repo"
  git -C "$LOCAL" archive $BASEC | ssh $HOST "tar -x -C $S/base"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BRANCH) > $S/repo/SYNCED_COMMIT; echo $BASEC > $S/base/SYNCED_COMMIT"
  echo "synced $(git -C "$LOCAL" rev-parse --short $BRANCH) + $BASEC to $HOST:$S"; exit ;;
start)
  st=${STAGES:-suite}; L=$S/go_${st// /_}.log
  ssh $HOST "STAGES='$st' GPU='${GPU:?set GPU}' O='${O:-}' nohup bash $S/repo/experiments/perf/combo/run.sh go > $L 2>&1 < /dev/null &
    echo started $L"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|go"; exit 1 ;;
esac

# ---- on 141 ----
. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel     # plugin model dir
BF16=/data41/audreymao/perf_20261030/bf16model                  # same model, bf16 weights
BATTN=/data41/audreymao/perf_20261030/repo/experiments/perf/det_compare/bench_attn.py
CONST=$A/data/intllama_const.pt
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-suite}; MAXWAIT=${MAXWAIT:-21600}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}.txt
: "${GPU:?set GPU}"
case " 2 3 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, branch $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 1800)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT' $log | tail -1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -3 | cut -c1-220; }
cmp() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
xcmp() {   # baseline run vs combo run: drop vLLM's seed-0 warm-up rows (not targets) from the baseline side
  grep -v '^0 ' $1.hash > $1.nowarm.hash; cp $1.json $1.nowarm.json; cmp $1.nowarm $2; }

suite() { D=$O/suite; mkdir -p $D
  for t in base repo; do
    wait_gpu || return 1
    echo "$(date +%T) start suite $t" | tee -a $SUM
    ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm GPUS=$GPU PY=$PY MODEL=$MODEL CONST=$CONST OUT=$D \
        timeout 36000 bash tests/vllm/run_tests.sh $t ) > $D/$t.out 2>&1
    echo "$(date +%T) end suite $t rc=$?" | tee -a $SUM
    grep -h "logit_hash_diff\|compare.*exit\|Traceback" $D/$t.out | sed "s|$D/||" | tee -a $SUM
  done
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    xcmp $D/base_$j $D/repo_$j
  done; }

bench() { D=$O/bench; mkdir -p $D
  for b in 1 32 128; do for t in base repo; do
    run $D/${t}_b${b}_t0.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp 0 --tag ${t}_b${b}_t0
  done; done
  grep -h "^RESULT" $D/*.log | cut -c1-260 | tee -a $SUM; }

bi() { D=$O/bi; mkdir -p $D
  for b in 1 32 128; do
    run $D/bi_triton_b$b.log 1800 repo env VLLM_BATCH_INVARIANT=1 $PY $BATTN --model $BF16 --batch $b --attn TRITON_ATTN
  done
  grep -h "^RESULT" $D/*.log | cut -c1-260 | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
