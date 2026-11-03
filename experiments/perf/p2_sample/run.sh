#!/usr/bin/env bash
# perf/elt2 (cheaper integer elementwise + logits/sampling at B=128) against 29e5cb5, one GPU, one run at a time.
#
# On the Mac:
#   bash experiments/perf/p2_sample/run.sh sync                    # git archive perf/elt2 -> $S/repo, 29e5cb5 -> $S/base
#   STAGES="smoke" GPU=1 bash experiments/perf/p2_sample/run.sh start
#   bash experiments/perf/p2_sample/run.sh status
# On 141: STAGES="..." GPU=n bash $S/repo/experiments/perf/p2_sample/run.sh go
#
# STAGES (TREES="base repo" by default; prof/bench/longctx run each tree in turn):
#   pytest   GPU pytest of the repo tree (tests/ -m gpu, plus tests/vllm/test_decode_paths.py, test_overflow_check.py)
#   smoke    base ref + alone_vllmgraph for both trees; compare.py and cmp of the hash files
#   suite    tests/vllm/run_tests.sh (14 jobs) for both trees; compare.py repo vs base and cmp of all 14 hash files
#   bench    tests/vllm/bench.py B=1/32/128 at T=0 and T=1, trees alternating per cell
#   longctx  tests/vllm/longctx_step.py S=16384, B=1 and 8
#   prof     experiments/perf/profile/prof_vllm.py B=128 (128 in / 128 out, T=0), then buckets.py report
# Outputs: $O/<stage>/..., $O/SUMMARY_<stages>.txt. Waits for GPU $GPU to be empty (0 MiB, no process) before each run.
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf2_sample
A=/data41/audreymao/a17_vllm
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
PROF_SRC=/Users/bytedance/Documents/projects/bitexact-p2-sample/../prof_tmp
BRANCH=${BRANCH:-perf2/sample}; BASEC=971a3e1
case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo $S/base && mkdir -p $S/repo $S/base"
  git -C "$LOCAL" archive $BRANCH | ssh $HOST "tar -x -C $S/repo"
  git -C "$LOCAL" archive $BASEC | ssh $HOST "tar -x -C $S/base"
  for t in repo base; do
    ssh $HOST "mkdir -p $S/$t/experiments/perf/profile"
    scp -q $PROF_SRC/prof_vllm.py $PROF_SRC/buckets.py $HOST:$S/$t/experiments/perf/profile/
  done
  ssh $HOST "echo $(git -C "$LOCAL" rev-parse --short $BRANCH) > $S/repo/SYNCED_COMMIT; echo $BASEC > $S/base/SYNCED_COMMIT"
  echo "synced $(git -C "$LOCAL" rev-parse --short $BRANCH) + $BASEC to $HOST:$S"; exit ;;
start)
  st=${STAGES:-smoke}; L=$S/go_${st// /_}.log
  ssh $HOST "STAGES='$st' TEMP='${TEMP:-0}' GPU='${GPU:?set GPU}' TREES='${TREES:-}' O='${O:-}' \
    nohup bash $S/repo/experiments/perf/p2_sample/run.sh go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -4 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

# ---- on 141 ----
. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel     # plugin model dir
CONST=$A/data/intllama_const.pt
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-smoke}; TREES=${TREES:-base repo}; MAXWAIT=${MAXWAIT:-21600}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}.txt
: "${GPU:?set GPU}"
:
echo "$(date +%T) GPU $GPU, stages: $STAGES, repo $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)" | tee -a $SUM

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

pytest() { D=$O/pytest; mkdir -p $D
  run $D/pytest.log 3600 repo env BITEXACT_INT8_CONST=$CONST $PY -m pytest -q -x tests/test_gpu_kernels.py \
    tests/test_vllm_kernels.py tests/test_int8_ops.py tests/vllm/test_decode_paths.py tests/vllm/test_overflow_check.py
  tail -3 $D/pytest.log | tee -a $SUM; }

smoke() { D=$O/smoke; mkdir -p $D
  run $D/base_ref.log 2400 base $PY tests/vllm/run_ref.py --const $CONST --out $D/base_ref
  for t in base repo; do
    run $D/${t}_alone_vllmgraph.log 2400 $t $PY tests/vllm/run_serve.py --model $MODEL --mode alone --prefix-cache 1 \
      --eager 0 --out $D/${t}_alone_vllmgraph
  done
  cmp_tree $D/base_ref $D/base_alone_vllmgraph $D/repo_alone_vllmgraph
  hcmp $D/base_alone_vllmgraph $D/repo_alone_vllmgraph; }

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
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    hcmp $D/base_$j $D/repo_$j
  done; }

bench() { D=$O/bench; mkdir -p $D
  for T in 0 1; do for b in 1 32 128; do for t in $TREES; do
    run $D/${t}_b${b}_t$T.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $T --tag ${t}_b${b}_t$T
  done; done; done
  grep -h "^RESULT" $D/*.log | cut -c1-260 | tee -a $SUM; }

longctx() { D=$O/longctx; mkdir -p $D
  for b in 1 8; do for t in $TREES; do
    run $D/${t}_s16k_b$b.log 1800 $t $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S 16384 --B $b
  done; done
  grep -h "^RESULT\|^{" $D/*.log | cut -c1-260 | tee -a $SUM; }

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 1800 $t $PY $S/repo/experiments/perf/p2_sample/tokens.py --model $MODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 1800 $t $PY $S/repo/experiments/perf/p2_sample/tokens.py --model $MODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 1800 $t $PY $S/repo/experiments/perf/p2_sample/tokens.py --model $MODEL --out $D/${t}_lp.json --logprobs
  done
  for k in mixed t0 lp; do cmp -s $D/base_$k.json $D/repo_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

prof() { D=$O/prof${TEMP:-0}; mkdir -p $D
  for t in $TREES; do
    mkdir -p $D/$t
    run $D/$t/int_b128.log 2400 $t bash -c "cd experiments/perf/profile && $PY prof_vllm.py --model $MODEL --batch 128 \
      --temp ${TEMP:-0} --tag ${t}_b128 --out $D/$t/int_b128"
    $PY $S/repo/experiments/perf/profile/buckets.py --report $D/$t > $D/$t/summary.md
    grep -h "^RESULT" $D/$t/int_b128.log | cut -c1-260 | tee -a $SUM
  done; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
