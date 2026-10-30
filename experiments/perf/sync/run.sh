#!/usr/bin/env bash
# Sampler-sync change (perf/sampler-sync) against the baseline commit 48a60f7, same GPU, one run at a time.
#
# On the Mac:
#   bash experiments/perf/sync/run.sh sync               # git archive of BRANCH -> $S/repo, of 48a60f7 -> $S/base
#   STAGES="smoke" bash experiments/perf/sync/run.sh start   # nohup on 141, log $S/go_<stages>.log
#   bash experiments/perf/sync/run.sh status
# On 141 (what "start" runs): STAGES="..." bash $S/repo/experiments/perf/sync/run.sh go
#
# STAGES (default "smoke"):
#   smoke    unit tests (kernels, sampler, decode paths) on the branch; sync_probe B=32 T=0/1 for both trees;
#            short ref + alone_vllmgraph + busy_vllmgraph_s5 for both trees, compared with the ref and each other
#   suite    tests/vllm/run_tests.sh (all 14 jobs, short + long) for both trees, then every branch run compared
#            with the same baseline run (logits hashes and tokens)
#   bench    tests/vllm/bench.py B=1/32/128 x T=0/1, both trees
#   prof     experiments/perf/profile/prof_vllm.py B=32/128 T=0 both trees (+ buckets.py summary per tree)
#   probe    sync_probe B=1/32/128 x T=0/1 both trees
# Env: GPU (fixed for the whole run; default: first of 0 1 7 that is free when the script starts),
#      MAXWAIT (s per run waiting for the GPU, default 21600).
# Every run waits until GPU $GPU shows 0 MiB used and no compute process, then runs under timeout.
# Outputs: $S/out/<stage>/<tree>_* (tree = base | br), $S/out/SUMMARY_<stages>.txt
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/speedup_20261030/sync
A=/data41/audreymao/a17_vllm
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
BRANCH=${BRANCH:-perf/sampler-sync}; BASEC=48a60f7

case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo $S/base && mkdir -p $S/repo $S/base $S/out"
  git -C "$LOCAL" archive $BRANCH | ssh $HOST "tar -x -C $S/repo"
  git -C "$LOCAL" archive $BASEC | ssh $HOST "tar -x -C $S/base"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BRANCH) > $S/repo/SYNCED_COMMIT; echo $BASEC > $S/base/SYNCED_COMMIT"
  echo "synced $(git -C "$LOCAL" rev-parse --short $BRANCH) + $BASEC to $HOST:$S"; exit ;;
start)
  st=${STAGES:-smoke}; L=$S/go_${st// /_}.log
  ssh $HOST "STAGES='$st' GPU='${GPU:-}' nohup bash $S/repo/experiments/perf/sync/run.sh go > $L 2>&1 < /dev/null &
    echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 25 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

# ---- on 141 ----
. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel     # plugin model dir (BitExactLlamaForCausalLM)
CONST=$A/data/intllama_const.pt
PROF=/data41/audreymao/perf_20261030/repo/experiments/perf/profile   # prof_vllm.py + buckets.py (not in git)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-smoke}; MAXWAIT=${MAXWAIT:-21600}
O=$S/out; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}.txt

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
if [ -z "${GPU:-}" ]; then
  w=0; while :; do for g in 0 1 7; do free $g && GPU=$g && break 2; done
    [ $w -ge $MAXWAIT ] && { echo "FAIL no free GPU"; exit 2; }; sleep 60; w=$((w + 60)); done
fi
case " 2 3 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, branch $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)" | tee -a $SUM

wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 1800)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm:$PROF CUDA_VISIBLE_DEVICES=$GPU \
      timeout $to "$@" ) > $log 2>&1; rc=$?
  echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed' $log | tail -1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -3 | cut -c1-220; }
cmp() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }

smoke() { D=$O/smoke; mkdir -p $D
  run $D/br_pytest.log 3600 repo env BITEXACT_INT8_CONST=$CONST $PY -m pytest -q tests/vllm/test_decode_paths.py \
    tests/test_vllm_kernels.py tests/test_sampling.py
  grep -h "passed\|failed" $D/br_pytest.log | tail -1 | sed 's/^/pytest branch: /' | tee -a $SUM
  for t in base repo; do
    for T in 0 1; do run $D/${t}_probe_b32_t$T.log 1800 $t $PY $S/repo/experiments/perf/sync/sync_probe.py \
      --model $MODEL --batch 32 --temp $T --tag ${t}_t$T; done
    run $D/${t}_ref.log 1800 $t $PY tests/vllm/run_ref.py --const $CONST --out $D/${t}_ref
    run $D/${t}_alone_vllmgraph.log 2400 $t $PY tests/vllm/run_serve.py --model $MODEL --mode alone \
      --prefix-cache 1 --eager 0 --out $D/${t}_alone_vllmgraph
    run $D/${t}_busy_vllmgraph_s5.log 2400 $t $PY tests/vllm/run_serve.py --model $MODEL --mode busy \
      --prefix-cache 1 --arrival-seed 5 --eager 0 --out $D/${t}_busy_vllmgraph_s5
  done
  grep -h "^SYNC\|^RESULT" $D/*_probe_*.log | cut -c1-260 | tee -a $SUM
  cmp $D/base_ref $D/repo_ref $D/repo_alone_vllmgraph $D/repo_busy_vllmgraph_s5 $D/base_alone_vllmgraph \
      $D/base_busy_vllmgraph_s5
  cmp $D/base_alone_vllmgraph $D/repo_alone_vllmgraph
  cmp $D/base_busy_vllmgraph_s5 $D/repo_busy_vllmgraph_s5; }

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
    ( cd $S/repo && $PY tests/vllm/compare.py $D/base_$j $D/repo_$j ) 2>&1 | sed "s|$D/||" | tee -a $SUM
  done; }

bench() { D=$O/bench; mkdir -p $D
  for b in 1 32 128; do for T in 0 1; do for t in base repo; do
    run $D/${t}_b${b}_t$T.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $T --tag ${t}_b${b}_t$T
  done; done; done
  grep -h "^RESULT" $D/*.log | cut -c1-260 | tee -a $SUM; }

prof() { for t in base repo; do D=$O/prof_$t; mkdir -p $D
    for b in 32 128; do run $D/int_b$b.log 2400 $t $PY $PROF/prof_vllm.py --model $MODEL --batch $b --tag ${t}_b$b \
      --out $D/int_b$b; done
    $PY $PROF/buckets.py --report $D > $D/summary.md
    grep -h "^TIMING\|^RESULT" $D/*.log | cut -c1-400 | tee -a $SUM; head -12 $D/summary.md | tee -a $SUM
  done; }

probe() { D=$O/probe; mkdir -p $D
  for b in 1 32 128; do for T in 0 1; do for t in base repo; do
    run $D/${t}_b${b}_t$T.log 1800 $t $PY $S/repo/experiments/perf/sync/sync_probe.py --model $MODEL --batch $b \
      --temp $T --tag ${t}_b${b}_t$T
  done; done; done
  grep -h "^SYNC\|^RESULT" $D/*.log | cut -c1-260 | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
