#!/usr/bin/env bash
# p14_verify: combined verification of p10-p13 on 141. base = 6faf8ef (origin/main), repo = e74549f (main + p10 + stacked GEMM).
#   bash run.sh sync            export both trees with git archive, copy this driver to 141
#   bash run.sh start           nohup the driver on 141 (STAGES default "corr bench prof")
#   bash run.sh status          driver summary tail and GPU state
#   bash run.sh fetch           pull small summaries and RESULT lines into experiments/perf/p14_verify/fetched/
# Stages: corr = correctness gate (both trees, lanes on GPUs 0 1 2 3 7); bench = throughput (one quiet GPU, base/repo
# alternating, reversed order on even reps); prof = one decode profile per tree at 1B and 8B, B=128, T=0.
# The driver never changes a tree: trees are immutable git archives on 141.
set -u
HOST=audreymao@10.154.162.141
ROOT=/data41/audreymao/perf14_20261106/verify
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BASE_SHA=6faf8ef
REPO_SHA=e74549f
BENCH_GPU=${BENCH_GPU:-7}
case "${1:-}" in
sync)
  ssh $HOST "mkdir -p $ROOT/p14 $ROOT/out && rm -rf $ROOT/base $ROOT/repo && mkdir -p $ROOT/base $ROOT/repo" || exit 1
  git -C "$REPO" archive $BASE_SHA | ssh $HOST "tar -x -C $ROOT/base" || exit 1
  git -C "$REPO" archive $REPO_SHA | ssh $HOST "tar -x -C $ROOT/repo" || exit 1
  ssh $HOST "echo $BASE_SHA > $ROOT/base/SYNCED_COMMIT; echo $REPO_SHA > $ROOT/repo/SYNCED_COMMIT"
  scp -q "$HERE/run.sh" $HOST:$ROOT/p14/run.sh && echo synced; exit ;;
start)
  ssh $HOST "export STAGES='${STAGES:-corr bench prof}' BENCH_GPU='$BENCH_GPU'; nohup bash $ROOT/p14/run.sh go >> $ROOT/out/driver.log 2>&1 < /dev/null & echo started"; exit ;;
status)
  ssh $HOST "tail -n 25 $ROOT/out/SUMMARY.txt 2>/dev/null; tail -n 3 $ROOT/out/driver.log; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; uptime"; exit ;;
fetch)
  mkdir -p "$HERE/fetched"
  scp -q -r $HOST:$ROOT/out/SUMMARY.txt $HOST:$ROOT/out/driver.log "$HERE/fetched/" 2>/dev/null
  scp -q -r $HOST:$ROOT/out/corr $HOST:$ROOT/out/bench $HOST:$ROOT/out/prof "$HERE/fetched/" 2>/dev/null
  echo fetched; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|fetch"; exit 1 ;;
esac

# ---------------- remote part (runs on 141 under nohup) ----------------
. /data41/audreymao/a17_vllm/env.sh
PY=/data41/audreymao/a17_vllm/venv/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0
O=$ROOT/out; mkdir -p $O/corr $O/bench $O/prof
SUM=$O/SUMMARY.txt
EX1=/data41/audreymao/perf7_20261105/exactplugin/exactmodel          # exact mode, 1B (bf16 weights)
EX8=/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel        # exact mode, 8B
IN1=/data41/audreymao/perf_20261030/det_compare/intmodel             # int8 mode, 1B
IN8=/data41/audreymao/p3_8b/intmodel8b                               # int8 mode, 8B
BF1=/data41/audreymao/perf_20261030/bf16model                        # stock bf16, 1B
BF8=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
CE1=/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt
CE8=/data41/audreymao/xhw8b/out/const_8b_public.pt
CI1=/data41/audreymao/a17_vllm/data/intllama_const.pt
CI8=/data41/audreymao/p3_8b/int8_const_8b.pt
HF=/data41/audreymao/abc_rl/hf; DATA=/data41/audreymao/abc_rl/data
SUITE_JOBS="ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager"
MAXWAIT=${MAXWAIT:-21600}

free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { g=$1; w=0; until free $g; do
    [ $w -ge $MAXWAIT ] && { echo "$(date +%T) FAIL gpu$g busy ${MAXWAIT}s" >> $SUM; return 1; }
    [ $((w % 600)) = 0 ] && echo "$(date +%T) wait gpu$g" >> $O/quiet.log; sleep 30; w=$((w + 30)); done
  sleep 3; free $g || wait_gpu $g; }
# run LOG TIMEOUT GPU TREE CMD...   (cwd = tree root; retries only on vLLM memory-profiling races)
run() { log=$1; to=$2; g=$3; t=$4; shift 4
  for try in 1 2 3 4; do
    wait_gpu $g || return 1
    ( cd $ROOT/$t && PYTHONPATH=$ROOT/$t/src:$ROOT/$t/tests/vllm CUDA_VISIBLE_DEVICES=$g timeout $to "$@" ) > $log 2>&1
    rc=$?; grep -q "Error in memory profiling\|No available memory for the cache\|Not enough memory" $log || break
    echo "$(date +%T) retry $try $(basename $log)" >> $SUM; done
  echo "$(date +%T) end $(basename $log .log) rc=$rc" >> $SUM
  return $rc; }
hcmp() { if cmp -s $1 $2; then echo "cmp OK $(basename $2)"; else echo "cmp DIFF $(basename $2)"; fi >> $SUM; }

pytest_t() { t=$1; g=$2
  run $O/corr/pytest_$t.log 7200 $g $t env BITEXACT_INT8_CONST=$CI1 BITEXACT_INT8_CONST_8B=$CI8 BITEXACT_EXACT_CONST=$CE1 \
      BITEXACT_MODEL=$BF1 BITEXACT_VLLM_MODEL=$IN1 $PY -m pytest -q -rs tests/
  echo "pytest $t: $(tail -n 1 $O/corr/pytest_$t.log)" >> $SUM; }

tokens_t() { t=$1; g=$2; D=$O/corr/tok; mkdir -p $D
  for mode in exact int8; do
    if [ $mode = exact ]; then M=$EX1; else M=$IN1; fi
    run $D/${t}_${mode}_mixed.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $M --out $D/${t}_${mode}_mixed.json
    run $D/${t}_${mode}_greedy.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $M --out $D/${t}_${mode}_greedy.json --temps 0
    run $D/${t}_${mode}_logprobs.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $M --out $D/${t}_${mode}_logprobs.json --logprobs
  done
  echo "tokens $t done" >> $SUM; }
tokcmp() { D=$O/corr/tok; for mode in exact int8; do for j in mixed greedy logprobs; do
    if cmp -s $D/base_${mode}_$j.json $D/repo_${mode}_$j.json; then echo "tokens $mode $j IDENTICAL"; else echo "tokens $mode $j DIFF"; fi >> $SUM
  done; done; }

check8_t() { t=$1; g=$2; D=$O/corr/c8; mkdir -p $D
  run $D/${t}_ref.log 5400 $g $t $PY experiments/perf/p7_exactplugin/ref8.py --const $CE8 --out $D/${t}_ref
  for e in 0 1; do
    run $D/${t}_eager$e.log 5400 $g $t $PY experiments/perf/p7_exactplugin/batch_inv.py --model $EX8 --out $D/${t}_eager$e.json --eager $e
    run $D/${t}_cmp$e.log 600 $g $t $PY experiments/perf/p7_exactplugin/cmp8.py $D/${t}_ref.hash $D/${t}_eager$e.json.hash
    echo "check8 $t eager$e: $(tail -n 1 $D/${t}_cmp$e.log)" >> $SUM
  done; }

spec_t() { t=$1; g=$2; D=$O/corr/spec; mkdir -p $D
  for k in 0 3; do
    run $D/${t}_k$k.log 5400 $g $t $PY tests/vllm/spec_check.py --model $EX1 --work suite --k $k --out $D/${t}_k$k
  done; echo "spec $t done" >> $SUM; }

plug_t() { t=$1; g=$2; D=$O/corr/plug; W=$ROOT/work/$t; mkdir -p $D $W $ROOT/cache/$t $ROOT/tmp/$t
  wait_gpu $g || return 1
  ( cd $ROOT/$t/experiments/abc_rl && env PYTHONPATH=$ROOT/$t/src TRITON_CACHE_DIR=$ROOT/cache/$t/triton \
      TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/$t/inductor VLLM_CACHE_ROOT=$ROOT/cache/$t/vllm TMPDIR=$ROOT/tmp/$t \
      XDG_CACHE_HOME=$ROOT/cache/$t FLASHINFER_WORKSPACE_BASE=$ROOT/cache/$t VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY check_plug.py --hf $HF --const $CI1 --data $DATA --work $W --out $D/${t}_res ) > $D/${t}.log 2>&1
  echo "check_plug $t rc=$?" >> $SUM; }

# suite NAME MODEL CONST TREE GPU : tests/vllm/run_tests.sh, one job at a time on that GPU; outputs tagged TREE_NAME_job
suite() { name=$1; M=$2; C=$3; t=$4; g=$5; D=$O/corr/suite; mkdir -p $D
  wait_gpu $g || return 1
  ( cd $ROOT/$t && PYTHONPATH=$ROOT/$t/src:$ROOT/$t/tests/vllm GPUS=$g PY=$PY MODEL=$M CONST=$C OUT=$D \
      timeout 36000 bash tests/vllm/run_tests.sh ${t}_$name ) > $D/${t}_$name.out 2>&1
  echo "suite $t $name rc=$?: $(grep -h 'compare' $D/${t}_$name.out | tr '\n' ' ')" >> $SUM; }

corr() {
  ( pytest_t base 0; pytest_t repo 0; tokens_t base 0; tokens_t repo 0; check8_t base 0; check8_t repo 0;
    spec_t base 0; spec_t repo 0 ) &
  ( suite exact $EX1 $CE1 base 1 ) &
  ( suite exact $EX1 $CE1 repo 2 ) &
  ( suite int8 $IN1 $CI1 base 3; plug_t base 3 ) &
  ( suite int8 $IN1 $CI1 repo 7; plug_t repo 7 ) &
  wait
  D=$O/corr/suite
  for name in exact int8; do for j in $SUITE_JOBS; do
    hcmp $D/base_${name}_$j.hash $D/repo_${name}_$j.hash
  done; done
  tokcmp
  echo "corr done $(date +%T)" >> $SUM; }

# one bench run; cmd built per mode
cell() { rep=$1; ord=$2; name=$3; M=$4; B=$5; T=$6; mode=$7; D=$O/bench; mkdir -p $D
  for t in $ord; do
    tag=${name}_B${B}_T${T}_${t}_r${rep}
    if [ $mode = bi ]; then cmd=(env VLLM_BATCH_INVARIANT=1 $PY tests/vllm/bench.py --attn TRITON_ATTN); else cmd=($PY tests/vllm/bench.py); fi
    run $D/$tag.log 2400 $BENCH_GPU $t "${cmd[@]}" --model $M --batch $B --temp $T --tag $tag
  done; }

bench() { for rep in 1 2 3 4 5; do
    if [ $((rep % 2)) = 1 ]; then ORD="repo base"; else ORD="base repo"; fi   # odd reps reversed (repo first)
    for B in 1 32 128; do
      cell $rep "$ORD" e1 $EX1 $B 0 plain
      cell $rep "$ORD" e1 $EX1 $B 1 plain
      cell $rep "$ORD" e8 $EX8 $B 0 plain
      cell $rep "repo" i1 $IN1 $B 0 plain
      cell $rep "repo" i8 $IN8 $B 0 plain
      cell $rep "repo" f1 $BF1 $B 0 plain
      cell $rep "repo" f8 $BF8 $B 0 plain
      cell $rep "repo" n1 $BF1 $B 0 bi
      cell $rep "repo" n8 $BF8 $B 0 bi
    done
    echo "bench rep $rep done $(date +%T)" >> $SUM
  done; echo "bench done" >> $SUM; }

prof() { D=$O/prof; mkdir -p $D
  for t in base repo; do
    run $D/${t}_1b.log 3600 $BENCH_GPU $t $PY experiments/perf/p8_exactfuse/prof_vllm.py --model $EX1 --batch 128 --temp 0 --out $D/${t}_1b --tag ${t}_1b
    run $D/${t}_8b.log 3600 $BENCH_GPU $t $PY experiments/perf/p8_exactfuse/prof_vllm.py --model $EX8 --batch 128 --temp 0 --out $D/${t}_8b --tag ${t}_8b
  done; echo "prof done" >> $SUM; }

echo "$(date +%T) driver start: base $(cat $ROOT/base/SYNCED_COMMIT) repo $(cat $ROOT/repo/SYNCED_COMMIT) stages: $STAGES" >> $SUM
for st in $STAGES; do $st; done
echo "$(date +%T) ALLDONE" >> $SUM
