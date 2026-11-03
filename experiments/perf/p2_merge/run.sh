#!/usr/bin/env bash
# p2_merge: perf2/merge (attn+gemm+sample) vs base 971a3e1, one GPU, one run at a time, quiet-host check before each launch.
#   bash run.sh sync ; STAGES="pytest suite tokens longctx bench ref prof" GPU=0 bash run.sh start ; bash run.sh status
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf2_20261103/merge
A=/data41/audreymao/a17_vllm
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
PROF_SRC=/Users/bytedance/Documents/projects/bitexact-llm/experiments/perf/profile
BRANCH=${BRANCH:-perf2/merge}; BASEC=971a3e1
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
  scp -q "$LOCAL/experiments/perf/p2_merge/"{bench_attn.py,tokens.py,run.sh} $HOST:$S/ ; exit ;;
start)
  st=${STAGES:-pytest}; L=$S/go_${st// /_}.log
  ssh $HOST "STAGES='$st' GPU='${GPU:?set GPU}' O='${O:-}' nohup bash $S/run.sh go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -5 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
BF16=/data41/audreymao/perf_20261030/bf16model
CONST=$A/data/intllama_const.pt
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-pytest}; TREES="base repo"; MAXWAIT=${MAXWAIT:-21600}; MAXLOAD=${MAXLOAD:-8}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}.txt
: "${GPU:?set GPU}"
case " 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, repo $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
# GPU empty and 1-min load average below MAXLOAD; log the state before every launch
wait_gpu() { w=0; while :; do
    ld=$(cut -d' ' -f1 /proc/loadavg)
    if free $GPU && [ "$(echo "$ld < $MAXLOAD" | bc)" = 1 ]; then break; fi
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy/load for ${MAXWAIT}s"; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait GPU $GPU load $ld"; sleep 30; w=$((w + 30)); done
  echo "$(date +%T) pre-launch: gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') allgpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',') load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $O/quiet_check.log; }
run() { log=$1; to=$2; tree=$3; shift 3
  wait_gpu || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed' $log | tail -1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -3 | cut -c1-220; }
cmp_tree() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }

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
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 \
           busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    hcmp $D/base_$j $D/repo_$j
  done; }

tokens() { D=$O/tokens; mkdir -p $D
  for t in $TREES; do
    run $D/${t}_mixed.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_mixed.json
    run $D/${t}_t0.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_t0.json --temps 0
    run $D/${t}_lp.log 1800 $t $PY $S/tokens.py --model $MODEL --out $D/${t}_lp.json --logprobs
  done
  for k in mixed t0 lp; do cmp -s $D/base_$k.json $D/repo_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

longctx() { D=$O/longctx; mkdir -p $D
  for b in 1 8; do for t in $TREES; do
    run $D/${t}_s16k_b$b.log 1800 $t $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S 16384 --B $b
  done; done
  grep -H "^RESULT\|^{" $D/*.log | cut -c1-300 | tee -a $SUM; }

bench() { D=$O/bench; mkdir -p $D
  for T in 0 1; do for b in 1 32 128; do for rep in 1 2 3 4 5; do for t in $TREES; do
    run $D/${t}_b${b}_t${T}_r$rep.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $T --tag ${t}_b${b}_t${T}_r$rep
  done; done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

ref() { D=$O/ref; mkdir -p $D
  for rep in 1 2 3 4 5; do for b in 1 32 128; do
    run $D/vllm_bi_b${b}_r$rep.log 1800 repo env VLLM_BATCH_INVARIANT=1 $PY $S/bench_attn.py --model $BF16 --batch $b --attn TRITON_ATTN
  done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

prof() { for T in 0 1; do D=$O/prof$T; mkdir -p $D
  for rep in 1 2 3; do for t in $TREES; do
    mkdir -p $D/$t$rep
    run $D/$t$rep/int_b128.log 2400 $t bash -c "cd experiments/perf/profile && $PY prof_vllm.py --model $MODEL --batch 128 \
      --temp $T --tag ${t}_b128 --out $D/$t$rep/int_b128"
    $PY $S/repo/experiments/perf/profile/buckets.py --report $D/$t$rep > $D/$t$rep/summary.md
    grep -H "^RESULT" $D/$t$rep/int_b128.log | cut -c1-300 | tee -a $SUM
  done; done; done; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
