#!/usr/bin/env bash
# p4_gemm8b (from p3_merge): perf3/merge vs base cc523a8 (main), one GPU per `go` process, one run at a time, GPU-empty check and
# host load logged before every launch (quiet_check.log).
#   bash run.sh sync ; STAGES="pytest suite" TREES=repo GPU=0 bash run.sh start ; bash run.sh status
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/p4_gemm8b
A=/data41/audreymao/a17_vllm
S8=/data41/audreymao/p3_8b
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
PROF_SRC=/Users/bytedance/Documents/projects/bitexact-llm/experiments/perf/profile
BRANCH=${BRANCH:-perf4/gemm8b}; BASEC=6d42e87
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
  scp -q "$LOCAL/experiments/perf/p3_merge/"{bench_attn.py,tokens.py} "$LOCAL/experiments/perf/p4_gemm8b/run.sh" $HOST:$S/ ; exit ;;
start)
  st=${STAGES:-pytest}; L=$S/go_${st// /_}_gpu${GPU:?set GPU}.log
  ssh $HOST "STAGES='$st' GPU='$GPU' TREES='${TREES:-}' O='${O:-}' nohup bash $S/run.sh go > $L 2>&1 < /dev/null & echo started $L"; exit ;;
status)
  ssh $HOST "tail -n 3 $S/go_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; uptime"; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|go"; exit 1 ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
MODEL=/data41/audreymao/perf_20261030/det_compare/intmodel
BF16=/data41/audreymao/perf_20261030/bf16model
CONST=$A/data/intllama_const.pt
HF8=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
C8=$S8/int8_const_8b.pt; M8=$S8/intmodel8b
export VLLM_ENABLE_V1_MULTIPROCESSING=0
STAGES=${STAGES:-pytest}; TREES=${TREES:-base repo}; MAXWAIT=${MAXWAIT:-86400}
O=${O:-$S/out}; mkdir -p $O; SUM=$O/SUMMARY_${STAGES// /_}_gpu$GPU.txt
: "${GPU:?set GPU}"
case " 4 5 6 " in *" $GPU "*) echo "GPU $GPU belongs to others"; exit 2;; esac
echo "$(date +%T) GPU $GPU, stages: $STAGES, trees: $TREES, repo $(cat $S/repo/SYNCED_COMMIT), base $(cat $S/base/SYNCED_COMMIT)" | tee -a $SUM

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { w=0; while ! free $GPU; do
    [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $GPU busy for ${MAXWAIT}s"; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait GPU $GPU"; sleep 30; w=$((w + 30)); done
  sleep 3; free $GPU || { wait_gpu; return; }
  echo "$(date +%T) $1 gpu${GPU}=$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') allgpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',') uptime=$(uptime | sed 's/.*load average: //')" >> $O/quiet_check.log; }
run() { log=$1; to=$2; tree=$3; shift 3       # run LOG TIMEOUT_S TREE CMD...   (cwd = tree root)
  wait_gpu $(basename $log .log) || return 1
  echo "$(date +%T) start $(basename $log .log)"
  ( cd $S/$tree && PYTHONPATH=$S/$tree/src:$S/$tree/tests/vllm CUDA_VISIBLE_DEVICES=$GPU timeout $to "$@" ) > $log 2>&1
  rc=$?; echo "$(date +%T) end $(basename $log .log) rc=$rc $(grep -h '^RESULT\|passed\|failed\|BATCH_INV' $log | tail -n 1 | cut -c1-220)"
  [ $rc = 0 ] || grep -h "Error\|error:" $log | tail -n 3 | cut -c1-220; }
cmp_tree() { echo "== $*" | tee -a $SUM; ( cd $S/repo && $PY tests/vllm/compare.py "$@" ) 2>&1 | tee -a $SUM; }
hcmp() { if cmp -s $1.hash $2.hash; then echo "cmp OK $(basename $2).hash"; else echo "cmp DIFF $(basename $2).hash"; fi | tee -a $SUM; }

pytest() { D=$O/pytest; mkdir -p $D
  run $D/pytest.log 3600 repo env BITEXACT_INT8_CONST=$CONST $PY -m pytest -q tests/test_gpu_kernels.py \
    tests/test_vllm_kernels.py tests/test_tri_fuse.py tests/test_int8_ops.py tests/vllm/test_decode_paths.py tests/vllm/test_overflow_check.py
  tail -n 3 $D/pytest.log | tee -a $SUM; }

suite() { D=$O/suite; mkdir -p $D
  for t in $TREES; do
    wait_gpu suite_$t || return 1
    echo "$(date +%T) start suite $t" | tee -a $SUM
    ( cd $S/$t && PYTHONPATH=$S/$t/src:$S/$t/tests/vllm GPUS=$GPU PY=$PY MODEL=$MODEL CONST=$CONST OUT=$D \
        timeout 36000 bash tests/vllm/run_tests.sh $t ) > $D/$t.out 2>&1
    echo "$(date +%T) end suite $t rc=$?" | tee -a $SUM
    grep -h "logit_hash_diff\|token_seqs_diff\|missing\|compare.*exit\|Traceback" $D/$t.out | sed "s|$D/||" | tee -a $SUM
  done; }
suitecmp() { D=$O/suite
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
  done; }
tokcmp() { D=$O/tokens
  for k in mixed t0 lp; do cmp -s $D/base_$k.json $D/repo_$k.json && echo "tokens $k IDENTICAL" || echo "tokens $k DIFF"; done | tee -a $SUM; }

# spec on (k=4) vs off (k=0): token identity, suite once + rep/bench x B 1,32 x T 0,0.7
spec() { D=$O/spec; mkdir -p $D; SC="$PY tests/vllm/spec_check.py --model $MODEL --eager 0"
  for k in 0 4; do run $D/suite_k$k.log 2400 repo $SC --work suite --k $k --out $D/suite_k$k; done
  for wk in rep bench; do for B in 1 32; do for T in 0 0.7; do for k in 0 4; do
    run $D/${wk}_B${B}_T${T}_k$k.log 2400 repo $SC --work $wk --batch $B --temp $T --k $k --out $D/${wk}_B${B}_T${T}_k$k
  done; done; done; done
  for f in $D/*_k0.json; do g=${f%_k0.json}_k4.json
    cmp -s $f $g && echo "spec $(basename ${f%_k0.json}) IDENTICAL" || echo "spec $(basename ${f%_k0.json}) DIFF"; done | tee -a $SUM; }

specbench() { D=$O/specbench; mkdir -p $D; SC="$PY tests/vllm/spec_check.py --model $MODEL --eager 0 --batch 1"
  for rep in 1 2 3; do for wk in rep bench; do for T in 0 0.7; do for k in 0 4; do
    run $D/${wk}_T${T}_k${k}_r$rep.log 2400 repo $SC --work $wk --temp $T --k $k --out $D/${wk}_T${T}_k${k}_r$rep
  done; done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

# 8B: plugin vs standalone IntLlama reference, 7 suite runs + batch invariance. GPU_UTIL 0.6 (16 GB weights).
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

longctx() { D=$O/longctx; mkdir -p $D
  for S_ in 16384 32768; do for b in 1 8; do for t in $TREES; do
    run $D/${t}_s${S_}_b$b.log 2400 $t $PY tests/vllm/longctx_step.py --const $CONST --model-dir $MODEL --S $S_ --B $b
  done; done; done
  grep -H "^RESULT\|^{" $D/*.log | cut -c1-300 | tee -a $SUM; }

# 1B bench: per rep, every cell: plugin base/repo at T=0,1 for B=1/32/128; then vLLM bf16 stock and BI+TRITON_ATTN (T=0)
bench() { D=$O/bench; mkdir -p $D
  for rep in 1 2 3 4 5; do
    for T in 0 1; do for b in 1 32 128; do for t in base repo; do
      run $D/${t}_b${b}_t${T}_r$rep.log 1800 $t $PY tests/vllm/bench.py --model $MODEL --batch $b --temp $T --tag ${t}_b${b}_t${T}_r$rep
    done; done; done
    for b in 1 32 128; do
      run $D/vllm_stock_b${b}_r$rep.log 1800 repo $PY $S/bench_attn.py --model $BF16 --batch $b
      run $D/vllm_bi_b${b}_r$rep.log 1800 repo env VLLM_BATCH_INVARIANT=1 $PY $S/bench_attn.py --model $BF16 --batch $b --attn TRITON_ATTN
    done
  done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

bench8() { D=$O/bench8; mkdir -p $D
  for rep in 1 2 3; do for b in 1 32 128; do
    run $D/plugin_b${b}_r$rep.log 2400 repo $PY tests/vllm/bench.py --model $M8 --batch $b --tag plugin8_b${b}_r$rep
    run $D/stock_b${b}_r$rep.log 2400 repo $PY $S/bench_attn.py --model $HF8 --batch $b
    run $D/bi_b${b}_r$rep.log 2400 repo env VLLM_BATCH_INVARIANT=1 $PY $S/bench_attn.py --model $HF8 --batch $b --attn TRITON_ATTN
  done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }

prof() { for T in 0 1; do D=$O/prof$T; mkdir -p $D
  for rep in 1 2 3; do for t in base repo; do
    mkdir -p $D/$t$rep
    run $D/$t$rep/int_b128.log 2400 $t bash -c "cd experiments/perf/profile && $PY prof_vllm.py --model $MODEL --batch 128 \
      --temp $T --tag ${t}_b128 --out $D/$t$rep/int_b128"
    $PY $S/repo/experiments/perf/profile/buckets.py --report $D/$t$rep > $D/$t$rep/summary.md
    grep -H "^RESULT" $D/$t$rep/int_b128.log | cut -c1-300 | tee -a $SUM
  done; done; done; }

bench8b() { D=$O/bench8b; mkdir -p $D
  for rep in 1 2 3; do for b in 1 32 128; do for t in base repo; do
    run $D/${t}_b${b}_r$rep.log 2400 $t $PY tests/vllm/bench.py --model $M8 --batch $b --tag ${t}8_b${b}_r$rep
  done; done; done
  grep -H "^RESULT" $D/*.log | cut -c1-300 | tee -a $SUM; }
prof8() { T=0; D=$O/prof8; mkdir -p $D; export GPU_UTIL=0.6
  for rep in 1 2 3; do for t in base repo; do mkdir -p $D/$t$rep
    run $D/$t$rep/int_b128.log 2400 $t bash -c "cd experiments/perf/profile && $PY prof_vllm.py --model $M8 --batch 128 --temp $T --tag ${t}_b128 --out $D/$t$rep/int_b128"
    $PY $S/repo/experiments/perf/profile/buckets.py --report $D/$t$rep > $D/$t$rep/summary.md
    grep -H "^RESULT" $D/$t$rep/int_b128.log | cut -c1-300 | tee -a $SUM
  done; done; unset GPU_UTIL; }

for st in $STAGES; do $st; done
echo "$(date +%T) done" | tee -a $SUM
