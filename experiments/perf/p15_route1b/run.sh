#!/usr/bin/env bash
# p15_route1b driver on 141, GPUs 0 and 1 only (one job per GPU).
#   bash run.sh sync     export base (eb1b09f) and repo (eb1b09f + working-tree changes) with git archive / tar
#   bash run.sh start    nohup the driver (STAGES default "holdout corr")
#   bash run.sh status   summary tail and GPU state
#   bash run.sh fetch    pull summaries into experiments/perf/p15_route1b/fetched/
set -u
HOST=audreymao@10.154.162.141
ROOT=/data41/audreymao/perf15_20261106/route1b
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BASE_SHA=eb1b09f
case "${1:-}" in
sync)
  ssh $HOST "mkdir -p $ROOT/p15 $ROOT/out && rm -rf $ROOT/base $ROOT/repo && mkdir -p $ROOT/base $ROOT/repo" || exit 1
  git -C "$REPO" archive $BASE_SHA | ssh $HOST "tar -x -C $ROOT/base" || exit 1
  git -C "$REPO" archive $BASE_SHA | ssh $HOST "tar -x -C $ROOT/repo" || exit 1
  (cd "$REPO" && git ls-files -m -o --exclude-standard src tests experiments/perf/p15_route1b) | \
    (cd "$REPO" && tar -cf - -T -) | ssh $HOST "tar -x -C $ROOT/repo" || exit 1
  ssh $HOST "echo $BASE_SHA > $ROOT/base/SYNCED_COMMIT; echo $BASE_SHA+worktree > $ROOT/repo/SYNCED_COMMIT; sha256sum $ROOT/repo/src/bitexact/vllm_plugin/exact_gemm.py | cut -c1-64 > $ROOT/out/repo_exact_gemm.sha256"
  scp -q "$HERE/run.sh" $HOST:$ROOT/p15/run.sh && echo synced; exit ;;
start)
  ssh $HOST "export STAGES='${STAGES:-holdout corr}'; nohup bash $ROOT/p15/run.sh go >> $ROOT/out/driver.log 2>&1 < /dev/null & echo started"; exit ;;
status)
  ssh $HOST "tail -n 30 $ROOT/out/SUMMARY.txt 2>/dev/null; tail -n 3 $ROOT/out/driver.log; nvidia-smi -i 0,1 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo"; exit ;;
fetch)
  mkdir -p "$HERE/fetched"
  scp -q $HOST:$ROOT/out/SUMMARY.txt "$HERE/fetched/" 2>/dev/null
  scp -q -r $HOST:$ROOT/out/corr $HOST:$ROOT/out/bench $HOST:$ROOT/out/prof $HOST:$ROOT/out/holdout.json "$HERE/fetched/" 2>/dev/null
  echo fetched; exit ;;
go) ;;
*) echo "usage: run.sh sync|start|status|fetch"; exit 1 ;;
esac

# ---------------- remote part (nohup on 141) ----------------
# one driver at a time: the lock is held for the life of this process and inherited by its children
mkdir -p $ROOT/out
exec 9>$ROOT/out/driver.lock
flock -n 9 || { echo "another p15 driver holds $ROOT/out/driver.lock; not starting" >&2; exit 1; }
. /data41/audreymao/a17_vllm/env.sh
PY=/data41/audreymao/a17_vllm/venv/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0
O=$ROOT/out; mkdir -p $O/corr $O/bench $O/prof
SUM=$O/SUMMARY.txt
EX1=/data41/audreymao/perf7_20261105/exactplugin/exactmodel
CE1=/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt
CI1=/data41/audreymao/a17_vllm/data/intllama_const.pt
CI8=/data41/audreymao/p3_8b/int8_const_8b.pt
BF1=/data41/audreymao/perf_20261030/bf16model
IN1=/data41/audreymao/perf_20261030/det_compare/intmodel
EX8=/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel
HF=/data41/audreymao/abc_rl/hf; DATA=/data41/audreymao/abc_rl/data
MAXWAIT=${MAXWAIT:-21600}
BENCH_GPU=${BENCH_GPU:-0}
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
env_c="BITEXACT_INT8_CONST=$CI1 BITEXACT_INT8_CONST_8B=$CI8 BITEXACT_EXACT_CONST=$CE1 BITEXACT_MODEL=$BF1 BITEXACT_VLLM_MODEL=$IN1"

holdout() { run $O/holdout.log 3600 0 repo $PY experiments/perf/p15_route1b/holdout.py --tree $ROOT/repo \
  --const $CE1 --rows 20,40,72,100,120 --out $O/holdout.json; echo "holdout done $(date +%T)" >> $SUM; }
pytest_t() { t=$1; g=$2
  run $O/corr/pytest_$t.log 7200 $g $t env $env_c $PY -m pytest -q -rs tests/
  echo "pytest $t: $(tail -n 1 $O/corr/pytest_$t.log)" >> $SUM; }
tokens_t() { t=$1; g=$2; D=$O/corr/tok; mkdir -p $D
  run $D/${t}_exact_mixed.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $EX1 --out $D/${t}_exact_mixed.json
  run $D/${t}_exact_greedy.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $EX1 --out $D/${t}_exact_greedy.json --temps 0
  run $D/${t}_exact_logprobs.log 1800 $g $t $PY experiments/perf/p4_merge/tokens.py --model $EX1 --out $D/${t}_exact_logprobs.json --logprobs
  echo "tokens $t done" >> $SUM; }
suite() { name=$1; M=$2; C=$3; t=$4; g=$5; D=$O/corr/suite; mkdir -p $D
  wait_gpu $g || return 1
  ( cd $ROOT/$t && PYTHONPATH=$ROOT/$t/src:$ROOT/$t/tests/vllm GPUS=$g PY=$PY MODEL=$M CONST=$C OUT=$D \
      timeout 36000 bash tests/vllm/run_tests.sh ${t}_$name ) > $D/${t}_$name.out 2>&1
  echo "suite $t $name rc=$?: $(grep -h 'compare' $D/${t}_$name.out | tr '\n' ' ')" >> $SUM; }

# GPU 0: holdout, pytest, tokens (sequential); GPU 1: exact 14-run suite (base then repo)
lane1() { suite exact $EX1 $CE1 base 1; suite exact $EX1 $CE1 repo 1; echo "lane1 done $(date +%T)" >> $SUM; }
echo "$(date +%T) driver start: base $(cat $ROOT/base/SYNCED_COMMIT) repo $(cat $ROOT/repo/SYNCED_COMMIT)" >> $SUM
if [[ " $STAGES " == *" holdout "* ]]; then holdout; fi
if [[ " $STAGES " == *" corr "* ]]; then
  lane1 &
  pytest_t base 0; pytest_t repo 0; tokens_t base 0; tokens_t repo 0
  wait
  D=$O/corr/suite
  for j in ref alone_pc1 alone_pc0 busy_pc1_s0 busy_pc0_s1 busy_pc1_s2_b64 busy_pc1_s3_preempt busy_eager_s4 busy_vllmgraph_s5 alone_vllmgraph long_ref long_alone long_busy long_busy_eager; do
    if [ -f $D/base_exact_$j.hash ] && cmp -s $D/base_exact_$j.hash $D/repo_exact_$j.hash; then echo "cmp OK exact_$j" >> $SUM; else echo "cmp DIFF exact_$j" >> $SUM; fi
  done
  for mode in mixed greedy logprobs; do if cmp -s $O/corr/tok/base_exact_$mode.json $O/corr/tok/repo_exact_$mode.json; then echo "tokens exact $mode IDENTICAL" >> $SUM; else echo "tokens exact $mode DIFF" >> $SUM; fi; done
  echo "corr done $(date +%T)" >> $SUM
fi

# throughput: 1B exact, B in 1 32 128, T=0, 5 reps; odd reps run repo first, even reps base first
cell() { rep=$1; ord=$2; B=$3; t_list=$ord; D=$O/bench; mkdir -p $D
  for t in $t_list; do tag=exact1_B${B}_T0_${t}_r${rep}
    run $D/$tag.log 2400 $BENCH_GPU $t $PY tests/vllm/bench.py --model $EX1 --batch $B --temp 0 --tag $tag
  done; }
bench_all() { for rep in 1 2 3 4 5; do if [ $((rep % 2)) = 1 ]; then ORD="repo base"; else ORD="base repo"; fi
    for B in 1 32 128; do cell $rep "$ORD" $B; done
    echo "bench rep $rep done $(date +%T)" >> $SUM; done; echo "bench done" >> $SUM; }
prof_all() { D=$O/prof; mkdir -p $D
  for t in base repo; do
    run $D/${t}_1b.log 3600 $BENCH_GPU $t $PY experiments/perf/p8_exactfuse/prof_vllm.py --model $EX1 --batch 128 --temp 0 --out $D/${t}_1b --tag ${t}_1b
  done; echo "prof done $(date +%T)" >> $SUM; }
# rerun of corr items that were killed by SIGTERM at 03:03 (base logprobs, base long_alone), then the long compare
rerun() { D=$O/corr/suite; T=$O/corr/tok
  run $T/base_exact_logprobs.log 1800 0 base $PY experiments/perf/p4_merge/tokens.py --model $EX1 --out $T/base_exact_logprobs.json --logprobs
  run $D/base_exact_long_alone.log 2400 1 base $PY tests/vllm/run_serve.py --model $EX1 --long --mode alone --prefix-cache 1 --eager 0 --out $D/base_exact_long_alone
  ( cd $ROOT/base && $PY tests/vllm/compare.py $D/base_exact_long_ref $D/base_exact_long_alone $D/base_exact_long_busy $D/base_exact_long_busy_eager ) > $D/base_long_compare.out 2>&1
  echo "rerun compare base long rc=$? $(grep -c 'token_seqs_diff 0' $D/base_long_compare.out) rows-ok" >> $SUM
  if cmp -s $T/base_exact_logprobs.json $T/repo_exact_logprobs.json; then echo "tokens exact logprobs IDENTICAL (rerun)" >> $SUM; else echo "tokens exact logprobs DIFF (rerun)" >> $SUM; fi
  echo "rerun done $(date +%T)" >> $SUM; }
if [[ " $STAGES " == *" rerun "* ]]; then rerun; fi
if [[ " $STAGES " == *" bench "* ]]; then bench_all; prof_all; fi
echo "$(date +%T) ALLDONE" >> $SUM
