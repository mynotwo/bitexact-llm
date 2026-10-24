#!/usr/bin/env bash
# Throughput (output tok/s, B requests of 128 input and 128 output tokens) on one GPU, one configuration at a time.
#   GPU=3 BF16_MODEL=DIR MODEL=DIR bash tests/vllm/bench_seq.sh
#   BF16_MODEL  plain Hugging Face Llama-3.2-1B directory (stock vLLM bf16)
#   MODEL       model directory from make_model_dir.py (integer plugin)
#   BATCHES     batch sizes (default: 1 32 128); OUT output directory (default: bench/)
# Each configuration starts only when its GPU has no memory in use. Results: one RESULT line per run in the log.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}; OUT=${OUT:-bench}; rm -rf "$OUT"; mkdir -p "$OUT"
: "${MODEL:?set MODEL}"; : "${BF16_MODEL:?set BF16_MODEL}"
P="$PY tests/vllm/bench.py"
busy() { nvidia-smi -i ${GPU:-0} --query-gpu=memory.used --format=csv,noheader,nounits; }
run() { name=$1; shift
  while [ "$(busy)" -gt 500 ]; do sleep 20; done
  CUDA_VISIBLE_DEVICES=${GPU:-0} timeout 3000 env "$@" > $OUT/$name.log 2>&1
  grep -h "^RESULT\|Error:" $OUT/$name.log | tail -1 | cut -c1-220 | sed "s/^/$name /"; }
for b in ${BATCHES:-1 32 128}; do
  run bf16_graph_b$b X=0 $P --model $BF16_MODEL --batch $b
  run bf16_eager_b$b X=0 $P --model $BF16_MODEL --batch $b --eager
  run bi_graph_b$b VLLM_BATCH_INVARIANT=1 $P --model $BF16_MODEL --batch $b
  run bi_eager_b$b VLLM_BATCH_INVARIANT=1 $P --model $BF16_MODEL --batch $b --eager
  run int_t0_b$b X=0 $P --model $MODEL --batch $b
  run int_t1_b$b X=0 $P --model $MODEL --batch $b --temp 1
  run int_eager_t0_b$b X=0 $P --model $MODEL --batch $b --eager
  run int_eager_t1_b$b X=0 $P --model $MODEL --batch $b --eager --temp 1
done
