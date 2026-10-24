#!/usr/bin/env bash
# Invariance suite for the vLLM plugin: short-prompt runs (12 targets) and long-context runs (5k, 9k, 14k tokens),
# each compared token by token and logits-hash by logits-hash with the integer reference run without vLLM.
#
#   GPUS="0 1 2" MODEL=DIR CONST=int8_const.pt bash tests/vllm/run_tests.sh TAG
#
#   MODEL  model directory from `python -m bitexact.vllm_plugin.make_model_dir` (vLLM runs)
#   CONST  constants file of the same model (reference runs)
#   GPUS   GPUs to use, one run per GPU at a time; start them only when the GPU is empty
#   PY     python with vLLM 0.32 and this package installed (default: python)
#   OUT    output directory (default: runs/)
# Exit status: 0 when every run matches the reference.
set -u
cd "$(dirname "$0")/../.."
T=${1:-run}; read -ra G <<< "${GPUS:-0}"
PY=${PY:-python}; OUT=${OUT:-runs}; mkdir -p "$OUT"
: "${MODEL:?set MODEL}"; : "${CONST:?set CONST}"
R="$PY tests/vllm/run_serve.py --model $MODEL"
JOBS=(
  "ref|$PY tests/vllm/run_ref.py --const $CONST"
  "alone_pc1|$R --mode alone --prefix-cache 1"
  "alone_pc0|$R --mode alone --prefix-cache 0"
  "busy_pc1_s0|$R --mode busy --prefix-cache 1 --arrival-seed 0"
  "busy_pc0_s1|$R --mode busy --prefix-cache 0 --arrival-seed 1"
  "busy_pc1_s2_b64|$R --mode busy --prefix-cache 1 --arrival-seed 2 --budget 64"
  "busy_pc1_s3_preempt|$R --mode busy --prefix-cache 1 --arrival-seed 3 --budget 512 --kv-blocks 160"
  "busy_eager_s4|env BITEXACT_GRAPHS=0 BITEXACT_COMPILE=0 $R --mode busy --prefix-cache 1 --arrival-seed 4 --max-seqs 128"
  "busy_vllmgraph_s5|$R --mode busy --prefix-cache 1 --arrival-seed 5 --eager 0"
  "alone_vllmgraph|$R --mode alone --prefix-cache 1 --eager 0"
  "long_ref|$PY tests/vllm/run_ref.py --long --model-dir $MODEL --const $CONST"
  "long_alone|$R --long --mode alone --prefix-cache 1 --eager 0"
  "long_busy|$R --long --mode busy --prefix-cache 1 --arrival-seed 8 --budget 512 --eager 0"
  "long_busy_eager|$R --long --mode busy --prefix-cache 0 --arrival-seed 9 --budget 2048"
)
NS=10   # the first NS jobs use the short targets, the rest the long ones
lane() {   # lane L runs jobs L, L + n, L + 2n, ... on GPU G[L]
  for ((j = $1; j < ${#JOBS[@]}; j += ${#G[@]})); do
    name=${JOBS[$j]%%|*}; cmd=${JOBS[$j]#*|}
    CUDA_VISIBLE_DEVICES=${G[$1]} timeout 2400 $cmd --out $OUT/${T}_$name > $OUT/${T}_$name.log 2>&1
  done
}
for ((l = 0; l < ${#G[@]}; l++)); do lane $l & done; wait
grep -l "Traceback" $OUT/${T}_*.log
REF=$OUT/${T}_ref
$PY tests/vllm/compare.py $REF $(for j in "${JOBS[@]:0:$NS}"; do echo $OUT/${T}_${j%%|*}; done)
echo compare_exit $?
REFL=$OUT/${T}_long_ref
$PY tests/vllm/compare.py $REFL $(for j in "${JOBS[@]:$NS}"; do echo $OUT/${T}_${j%%|*}; done)
echo compare_long_exit $?
$PY tests/vllm/plan_stats.py $(for j in "${JOBS[@]:3:7}" "${JOBS[@]:11}"; do echo $OUT/${T}_${j%%|*}; done)
for f in $OUT/${T}_busy_*.log $OUT/${T}_long_busy*.log; do echo "$f $(grep -h '^preemptions\|^metric vllm:prefix_cache_hits' $f | tr '\n' ' ')"; done
