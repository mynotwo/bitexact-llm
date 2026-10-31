#!/usr/bin/env bash
# Baseline (48a60f7) vs this branch on 141, same GPU: full invariance suite, bench, profiler.
#
# On the Mac:
#   bash experiments/perf/fuse/run.sh sync      # git archive 48a60f7 -> $S/base_repo, branch HEAD -> $S/repo,
#                                               # plus experiments/perf/profile (untracked in the main checkout)
#   bash experiments/perf/fuse/run.sh start     # nohup "go" on 141, log $S/go.log
#   bash experiments/perf/fuse/run.sh status
# On 141: bash $S/repo/experiments/perf/fuse/run.sh go [STAGES]   (default: suite bench prof hashes summary)
#   suite   tests/vllm/run_tests.sh (short + long) for base and branch -> $O/suite/{base,fuse}_*
#   hashes  compare.py: branch runs (incl. its own run_ref) against the BASELINE reference run_ref
#   bench   tests/vllm/bench.py B=1/32/128 x T=0/1, base and branch alternating -> $O/bench/*.log
#   prof    experiments/perf/profile/prof_vllm.py B=32/128 + buckets.py --report -> $O/prof/{base,fuse}/
# One GPU G (first free of $GPUS, never 2-6) is chosen once; before every run G must show 0 MiB and no
# compute process, else the run waits (no time limit on waiting, MAXWAIT s then FAIL). Each run has a timeout.
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/speedup_20261030/fuse
LOCAL=/Users/bytedance/Documents/projects/bitexact-llm
WT=/Users/bytedance/Documents/projects/bitexact-speedup-fuse
case "${1:-}" in
sync)
  ssh $HOST "rm -rf $S/repo $S/base_repo && mkdir -p $S/repo $S/base_repo"
  git -C $LOCAL archive 48a60f7 | ssh $HOST "tar -x -C $S/base_repo"
  git -C $WT archive HEAD | ssh $HOST "tar -x -C $S/repo"
  for d in base_repo repo; do COPYFILE_DISABLE=1 tar -C $LOCAL -c experiments/perf/profile/buckets.py \
    experiments/perf/profile/prof_vllm.py | ssh $HOST "tar -x -C $S/$d"; done
  ssh $HOST "echo 48a60f7 > $S/base_repo/SYNCED_COMMIT; echo $(git -C $WT rev-parse --short HEAD) > $S/repo/SYNCED_COMMIT"
  exit ;;
start) ssh $HOST "nohup bash $S/repo/experiments/perf/fuse/run.sh go ${2:-} > $S/go.log 2>&1 < /dev/null & echo started"; exit ;;
status) ssh $HOST "tail -8 $S/go.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"; exit ;;
go) shift ;;
*) echo "usage: run.sh sync|start|status|go [STAGES]"; exit 1 ;;
esac

# ---- on 141 ----
A=/data41/audreymao/a17_vllm; PY=$A/venv/bin/python; O=$S/cmp
. $A/env.sh                                    # its PYTHONPATH is replaced per run below
export VLLM_ENABLE_V1_MULTIPROCESSING=0 TORCHINDUCTOR_CACHE_DIR=$S/inductor_cache_cmp
CONST=$A/data/intllama_const.pt; IM=/data41/audreymao/perf_20261030/det_compare/intmodel
GPUS=${GPUS:-0 1 7}; MAXWAIT=${MAXWAIT:-28800}
declare -A R=([base]=$S/base_repo [fuse]=$S/repo)
mkdir -p $O/suite $O/bench $O/prof/base $O/prof/fuse
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
G=""; w=0
while [ -z "$G" ]; do
  for g in $GPUS; do case $g in 2|3|4|5|6) continue;; esac; free $g && { G=$g; break; }; done
  [ -z "$G" ] && { [ $w -ge $MAXWAIT ] && { echo "FAIL no free GPU"; exit 9; }; sleep 30; w=$((w + 30)); }
done
echo "$(date +%T) GPU $G: $(nvidia-smi -i $G --query-gpu=name --format=csv,noheader); base $(cat $S/base_repo/SYNCED_COMMIT) fuse $(cat $S/repo/SYNCED_COMMIT)"
waitg() { w=0; while ! free $G; do [ $w -ge $MAXWAIT ] && { echo "FAIL GPU $G busy"; exit 9; }; sleep 30; w=$((w + 30)); done; }
run() { v=$1 log=$2 to=$3; shift 3; waitg      # run VARIANT LOG TIMEOUT CMD...
  echo "$(date +%T) start $(basename $log)"
  (cd ${R[$v]} && CUDA_VISIBLE_DEVICES=$G PYTHONPATH=${R[$v]}/src:${R[$v]}/tests/vllm:${R[$v]}/experiments/perf/profile timeout $to "$@") > $log 2>&1
  echo "$(date +%T) end rc=$? $(grep -h '^RESULT\|compare_exit\|compare_long_exit' $log | tail -2 | cut -c1-200)"; }

suite() {    # run_tests.sh itself starts jobs without checking the GPU: waitg before the whole suite only
  for v in base fuse; do
    run $v $O/suite/$v.txt 21600 env GPUS=$G MODEL=$IM CONST=$CONST PY=$PY OUT=$O/suite bash tests/vllm/run_tests.sh $v
    grep -h "compare\|logit_hash_diff" $O/suite/$v.txt | cut -c1-200
  done; }
hashes() {   # branch against the baseline reference (and base vs its own reference, from the suite output)
  cd $S/repo
  { $PY tests/vllm/compare.py $O/suite/base_ref $O/suite/fuse_ref $O/suite/fuse_alone_pc1 $O/suite/fuse_alone_pc0 \
      $O/suite/fuse_busy_pc1_s0 $O/suite/fuse_busy_pc0_s1 $O/suite/fuse_busy_pc1_s2_b64 $O/suite/fuse_busy_pc1_s3_preempt \
      $O/suite/fuse_busy_eager_s4 $O/suite/fuse_busy_vllmgraph_s5 $O/suite/fuse_alone_vllmgraph; echo "short_vs_base_exit $?"
    $PY tests/vllm/compare.py $O/suite/base_long_ref $O/suite/fuse_long_ref $O/suite/fuse_long_alone \
      $O/suite/fuse_long_busy $O/suite/fuse_long_busy_eager; echo "long_vs_base_exit $?"
    for f in ref long_ref; do cmp -s $O/suite/base_$f.hash $O/suite/fuse_$f.hash && echo "$f.hash identical" || echo "$f.hash DIFFERS"; done
  } > $O/hashes.txt 2>&1; cat $O/hashes.txt; }
bench() {
  for b in 1 32 128; do for t in 0 1; do for v in base fuse; do
    run $v $O/bench/${v}_t${t}_b$b.log 1800 $PY tests/vllm/bench.py --model $IM --batch $b --temp $t
  done; done; done; }
prof() {
  for b in 32 128; do for v in base fuse; do
    run $v $O/prof/$v/int_b$b.log 2400 $PY experiments/perf/profile/prof_vllm.py --model $IM --batch $b \
      --tag int_b$b --out $O/prof/$v/int_b$b
  done; done
  for v in base fuse; do $PY ${R[$v]}/experiments/perf/profile/buckets.py --report $O/prof/$v > $O/prof/$v/summary.md; done; }
summary() {
  { echo "GPU $G base $(cat $S/base_repo/SYNCED_COMMIT) fuse $(cat $S/repo/SYNCED_COMMIT)"
    echo "== suite"; grep -h "compare_exit\|compare_long_exit\|Traceback" $O/suite/base.txt $O/suite/fuse.txt
    grep -h "logit_hash_diff" $O/suite/fuse.txt | awk '{print $1, $5, $9, $11}'
    echo "== hashes vs baseline"; cat $O/hashes.txt
    echo "== bench tok/s"; for f in $O/bench/*.log; do echo "$(basename $f .log) $(grep -h '^RESULT' $f | grep -o '"tok_per_s": [0-9.]*')"; done
    echo "== prof decode ms/step"; for v in base fuse; do echo "-- $v"; sed -n '1,12p' $O/prof/$v/summary.md; done
  } > $O/SUMMARY.txt 2>&1; cat $O/SUMMARY.txt; }

for s in ${1:-suite hashes bench prof summary}; do
  case $s in suite|hashes|bench|prof|summary) $s ;; *) echo "unknown stage $s"; exit 2 ;; esac
done
echo ALLDONE
