#!/bin/bash
# Portability runs on 138 (8x H100). Usage: run138.sh LANE GPUS   (GPUS comma list, each must show 0 MiB)
# Lanes: ref (1 GPU), tp (4 GPUs), ver (1 GPU), cpu (none). Resumable: a step is skipped when its output exists.
# Workload everywhere: 32 GSM8K prompts x G 8 (seeds 1 + 8p + g, the first 256 of abc_measure Q1), <= 256 new tokens.
. $HOME/portability/env138.sh; cd $P
LANE=$1; export CUDA_VISIBLE_DEVICES=$2
WL="--data $DATA --n-prompts 32 --g 8"
waitgpu() { [ -z "$CUDA_VISIBLE_DEVICES" ] && return; while true; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v L=",$CUDA_VISIBLE_DEVICES," 'index(L, ","$1",") && $2>0' | wc -l)
  [ "$busy" = 0 ] && return; echo "wait gpus $CUDA_VISIBLE_DEVICES busy $(date +%T)"; sleep 60; done; }
waitf() { until [ -f "$1" ]; do sleep 30; done; }
step() { out=$1; shift; if [ -f "$out" ]; then echo "SKIP $out"; return; fi; waitgpu; echo "RUN $out $(date +%T)"
  timeout 2400 "$@" > $out.log 2>&1 && echo "OK $out $(date +%T)" || { echo "FAIL $out $(date +%T): $(grep -E 'Error|error' $out.log | tail -1 | cut -c1-200)"; touch $out.failed; }; }
BI="env VLLM_BATCH_INVARIANT=1"
dsamp() { py=$1; out=$2; shift 2; step $O/$out.json $BI $py $X/sample.py --engine bf16 --attn TRITON_ATTN --model $DEQ $WL --out $O/$out.json "$@"; }
dscore() { py=$1; out=$2; roll=$3; shift 3; waitf $O/$roll.json; step $O/$out.json $BI $py $X/score_d.py --attn TRITON_ATTN --rollout $O/$roll.json --model $DEQ --data $DATA --max-seqs 128 --out $O/$out.json "$@"; }
csamp() { py=$1; out=$2; shift 2; step $O/$out.json $py $X/sample.py --engine bitexact --model $INT $WL --out $O/$out.json "$@"; }
cscore() { py=$1; out=$2; roll=$3; shift 3; waitf $O/$roll.json; step $O/$out.json $py $R/bitexact_engine.py score --const $CONST --rollout $O/$roll.json --tokenizer $DEQ --data $DATA --out $O/$out.json "$@"; }
case $LANE in
ref)
  dsamp $PY D_t1; dsamp $PY D_t0 --temp 0; dscore $PY Ds_t1 D_t1
  csamp $PY C_t1; csamp $PY C_t0 --temp 0; cscore $PY Cs_t1 C_t1 --device cuda --bs 8
  cscore $PY Cs_h800B h800_q1_B --device cuda --bs 16          # C cross-GPU: H800 plugin rollouts scored on H100
  dsamp $PY D_t1_ms64 --max-seqs 64; dsamp $PY D_t1_ms1 --max-seqs 1
  dscore $PY Ds_t1_ms8 D_t1 --max-seqs 8
  csamp $PY C_t1_ms64 --max-seqs 64; csamp $PY C_t1_ms1 --max-seqs 1
  cscore $PY Cs_t1_bs1 C_t1 --device cuda --bs 1; cscore $PY Cs_t1_bs64 C_t1 --device cuda --bs 64 ;;
tp)
  for tp in 2 4; do
    dsamp $PY D_t1_tp$tp --tp $tp; dsamp $PY D_t0_tp$tp --tp $tp --temp 0
    dscore $PY Ds_t1_tp$tp D_t1 --tp $tp
    dsamp $PY D_t1_tp${tp}_ms64 --tp $tp --max-seqs 64; dsamp $PY D_t1_tp${tp}_ms1 --tp $tp --max-seqs 1
  done
  csamp $PY C_t1_tp2 --tp 2 ;;                                  # plugin under TP (expected unsupported)
ver)
  for v in ${VERS:-0.31.0 0.33.0}; do
    until grep -q "INSTALLED_$v\|ERROR" $P/install_$v.log 2>/dev/null; do sleep 60; done
    grep -q INSTALLED_$v $P/install_$v.log || { echo "INSTALL FAILED $v"; continue; }
    V=$P/venv_$v/bin/python
    dsamp $V D_t1_v$v; dsamp $V D_t0_v$v --temp 0; dscore $V Ds_t1_v$v D_t1
    csamp $V C_t1_v$v; csamp $V C_t0_v$v --temp 0; cscore $V Cs_t1_v$v C_t1 --device cuda --bs 8
  done ;;
cpu)
  waitf $O/C_t1.json; waitf $O/D_t1.json
  cscore $PY Cs_t1_cpu C_t1 --device cpu --bs 4 --threads 32 --seeds-from $X/seeds16.txt
  for dt in bfloat16 float32; do
    step $O/Ds_t1_cpu_hf_$dt.json $BI $PY $X/score_cpu_float.py --rollout $O/D_t1.json --model $DEQ --data $DATA --dtype $dt --seeds-from $X/seeds16.txt --batch-invariant --out $O/Ds_t1_cpu_hf_$dt.json
  done
  # vLLM CPU attempt with the installed (CUDA) wheel, no GPU visible
  step $O/Ds_t1_cpu_vllm.json env CUDA_VISIBLE_DEVICES= VLLM_BATCH_INVARIANT=1 $PY $X/score_d.py --rollout $O/D_t1.json --model $DEQ --data $DATA --max-seqs 8 --dtype bfloat16 --out $O/Ds_t1_cpu_vllm.json ;;
esac
echo "LANEDONE $LANE"
