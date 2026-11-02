#!/usr/bin/env bash
# perf/b128 (perf/attn + perf/elt2) against 29e5cb5, plus the pre-combo plugin 48a60f7 and vLLM bf16 references.
# Two lanes, one run per GPU at a time: lane "check" (pytest + suite) and lane "perf" (bench + longctx).
#
# On the Mac:
#   bash experiments/perf/b128/run.sh sync     # git archive perf/b128 -> $S/repo, 29e5cb5 -> $S/base, 48a60f7 -> $S/pre
#   STAGES="pytest suite" GPU=1 bash experiments/perf/b128/run.sh start
#   STAGES="bench longctx" GPU=7 bash experiments/perf/b128/run.sh start
# On 141: STAGES="..." GPU=n bash $S/repo/experiments/perf/b128/run.sh go
#
# STAGES:
#   pytest   GPU pytest of the repo tree (tests/ -m gpu files, test_decode_paths.py, test_overflow_check.py)
#   suite    tests/vllm/run_tests.sh (14 jobs) for base then repo; compare.py repo vs base_ref and cmp of the 14 hash files
#   bench    ROUNDS rounds; per round and batch (1/32/128): T=0 base/repo/pre (order rotates per round), vLLM bf16 stock
#            and VLLM_BATCH_INVARIANT=1 + TRITON_ATTN (bench_attn.py), then T=1 base/repo. Median per cell in SUMMARY.
#   longctx  tests/vllm/longctx_step.py S=16384, B=1 and 8, base and repo
# Outputs: $O/<stage>/..., $O/SUMMARY_<stages>.txt. Waits for GPU $GPU to be empty (0 MiB, no process) before each run.
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/b128_20261031/merge
A=/data41/audreymao/a17_vllm
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
BRANCH=${BRANCH:-perf/b128}; BASEC=29e5cb5; PREC=48a60f7
case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo $S/base $S/pre && mkdir -p $S/repo $S/base $S/pre"
  git -C "$LOCAL" archive $BRANCH | ssh $HOST "tar -x -C $S/repo"
  git -C "$LOCAL" archive $BASEC | ssh $HOST "tar -x -C $S/base"
  git -C "$LOCAL" archive $PREC | ssh $HOST "tar -x -C $S/pre"
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BRANCH) > $S/repo/SYNCED_COMMIT; echo $BASEC > $S/base/SYNCED_COMMIT;
    echo $PREC > $S/pre/SYNCED_COMMIT"
  echo "synced $(git -C "$LOCAL" rev-parse --short $BRANCH) + $BASEC + $PREC to $HOST:$S"; exit ;;
start)
  st=${STAGES:?set STAGES}; L=$S/go_${st// /_}.log
  ssh $HOST "STAGES='$st' GPU='${GPU:?set GPU}' ROUNDS='${ROUNDS:-}' O='${O:-}' \
    nohup bash $S/repo/experiments/perf/b128/run.sh go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -4 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

# ---- on 141 ----
. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel     # plugin model dir
BF16=/data41/audreymao/perf_20261030/bf16model                  # same model, bf16 weights
BATTN=/data41/audreymao/perf_20261030/repo/experiments/perf/det_compare/bench_attn.py
CONST=$A/data/intllama_const.pt
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:?set STAGES}; ROUNDS=${ROUNDS:-3}; MAXWAIT=${MAXWAIT:-21600}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}.txt
: "${GPU:?set GPU}"
case " 2 3 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, repo $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)," \
  "pre $(cat $S/pre/SYNCED_COMMIT)" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 1800)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed' $log | tail -1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -3 | cut -c1-220; }
cmp_tree() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }
JOBS="ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5
      alone_vllmgraph long_ref long_alone long_busy long_busy_eager"

pytest() { D=$O/pytest; mkdir -p $D
  run $D/pytest.log 3600 repo env BITEXACT_INT8_CONST=$CONST $PY -m pytest -q tests/test_gpu_kernels.py \
    tests/test_vllm_kernels.py tests/test_int8_ops.py tests/vllm/test_decode_paths.py tests/vllm/test_overflow_check.py
  tail -3 $D/pytest.log | tee -a $SUM; }

suite() { D=$O/suite; mkdir -p $D
  for t in base repo; do
    wait_gpu || return 1
    echo "$(date +%T) start suite $t" | tee -a $SUM
    ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm GPUS=$GPU PY=$PY MODEL=$MODEL CONST=$CONST OUT=$D \
        timeout 36000 bash tests/vllm/run_tests.sh $t ) > $D/$t.out 2>&1
    echo "$(date +%T) end suite $t rc=$?" | tee -a $SUM
    grep -h "logit_hash_diff\|token_seqs_diff\|missing\|compare.*exit\|Traceback" $D/$t.out | sed "s|$D/||" | tee -a $SUM
  done
  cmp_tree $D/base_ref $D/repo_ref $(for j in alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 \
    busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5 alone_vllmgraph; do echo $D/repo_$j; done)
  cmp_tree $D/base_long_ref $D/repo_long_ref $D/repo_long_alone $D/repo_long_busy $D/repo_long_busy_eager
  for j in $JOBS; do hcmp $D/base_$j $D/repo_$j; done; }

bench() { D=$O/bench; mkdir -p $D
  for r in $(seq 1 $ROUNDS); do for b in 1 32 128; do
    case $((r % 3)) in 1) ord="base repo pre";; 2) ord="repo pre base";; 0) ord="pre base repo";; esac
    for t in $ord; do
      run $D/${t}_b${b}_t0_r$r.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp 0 --tag ${t}_b${b}_t0
    done
    run $D/bf16_b${b}_r$r.log 1800 repo $PY $BATTN --model $BF16 --batch $b
    run $D/bi_triton_b${b}_r$r.log 1800 repo env VLLM_BATCH_INVARIANT=1 $PY $BATTN --model $BF16 --batch $b \
      --attn TRITON_ATTN
    for t in $( [ $((r % 2)) = 1 ] && echo "base repo" || echo "repo base"); do
      run $D/${t}_b${b}_t1_r$r.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp 1 --tag ${t}_b${b}_t1
    done
  done; done
  echo "-- tok/s per cell: runs, median" | tee -a $SUM
  for c in base_b{1,32,128}_t0 repo_b{1,32,128}_t0 pre_b{1,32,128}_t0 bf16_b{1,32,128} bi_triton_b{1,32,128} \
           base_b{1,32,128}_t1 repo_b{1,32,128}_t1; do
    v=$(for f in $D/${c}_r*.log; do grep -h '^RESULT' $f | tail -1 | grep -o '"tok_per_s": [0-9.]*' | cut -d' ' -f2; done)
    echo "$c $(echo $v) median $(echo "$v" | sort -n | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}')"
  done | tee -a $SUM; }

longctx() { D=$O/longctx; mkdir -p $D
  for b in 1 8; do for t in base repo; do
    run $D/${t}_s16k_b$b.log 1800 $t $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S 16384 --B $b
  done; done
  for f in $D/*.log; do echo "$(basename $f .log) $(grep -h '^RESULT\|^{' $f | tail -1 | cut -c1-240)"; done | tee -a $SUM; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
