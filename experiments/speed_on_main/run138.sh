#!/usr/bin/env bash
# speed-on-main gates and speed on 138 (H100). Code trees are immutable copies per tag: ~/som/code_TAG (git archive);
# the harness (this directory) runs from ~/som/harness for every tag, so the base tree (origin/main) needs no copy of it.
#   run138.sh sync TAG REV                 (run on the Mac: copies commit REV to 138, and this directory to ~/som/harness)
#   run138.sh pytest TAG G                 full test suite (BITEXACT_INT8_CONST set)
#   run138.sh suite TAG "G1 G2 G3"         main's invariance suite (tests/vllm/run_tests.sh)
#   run138.sh scheck TAG G                 scorer gate, 8 configurations       run138.sh stime TAG G   scorer tok/s
#   run138.sh gate TAG G1 G2 G3 G4         decode rollouts T=1/greedy x B=1/64  (then: gate.py BASE TAG)
#   run138.sh dspeed TAG G                 decode ms/step, B = 1, 64 (512 tokens), 512 (256 tokens), median of 3
#   run138.sh bench TAG G                  main's README throughput (tests/vllm/bench.py, int8 T=0/1, B 1/32/128), 3 reps
# Every GPU must show 0 MiB at launch.
MODE=$1; TAG=$2; shift 2
H=audreymao@10.73.164.138
if [ $MODE = sync ]; then
  R="$(cd $(dirname $0)/../.. && pwd)"
  ssh -n -o ConnectTimeout=10 $H "rm -rf ~/som/code_$TAG; mkdir -p ~/som/code_$TAG ~/som/out"
  git -C $R archive $1 | ssh -o ConnectTimeout=10 $H "tar -x -C ~/som/code_$TAG"
  rsync -a --exclude __pycache__ $R/experiments/speed_on_main/ $H:som/harness/; exit; fi
W=$HOME/abc_measure; S=$HOME/som; O=$S/out; C=$S/code_$TAG; X=$S/harness
PY=$W/venv/bin/python; CONST=$W/const.pt; DEQ=$W/deqmodel; DATA=$W/gsm8k; INT=$HOME/portability/intmodel
GSM=$HOME/profile/roll/q1_B.json; L4K=$HOME/profile/roll/q2_4096_B.json; REF=$HOME/speed2/out
export PYTHONPATH=$C/src:$C/experiments/rl_mismatch:$X TMPDIR=$S/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME=$S/cache TRITON_CACHE_DIR=$S/cache/triton VLLM_CACHE_ROOT=$S/cache/vllm \
  TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor BITEXACT_INT8_CONST=$CONST
mkdir -p $O $S/tmp $S/cache; cd $C
chk() { for g in "$@"; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" = 0 ] || { echo "GPU $g busy ($u MiB)"; exit 1; }; done; }
for g in $*; do chk $g; done
B="$PY $X/scorer_bench.py --const $CONST --model $DEQ --data $DATA"
r() { "$@" 2>&1 | grep '^{' | tee -a $J || true; }
case $MODE in
pytest) CUDA_VISIBLE_DEVICES=$1 $PY -m pytest -q -p no:cacheprovider tests > $O/${TAG}_pytest.log 2>&1; tail -n 2 $O/${TAG}_pytest.log ;;
suite) GPUS="$1" PY=$PY MODEL=$INT CONST=$CONST OUT=$O/suite_$TAG bash tests/vllm/run_tests.sh $TAG > $O/${TAG}_suite.log 2>&1
  grep -h "compare_exit\|compare_long_exit\|token_seqs_diff\|Traceback" $O/${TAG}_suite.log | head -40 ;;
scheck) export CUDA_VISIBLE_DEVICES=$1; J=$O/$TAG.scheck.jsonl; : > $J; M="--mode ${SMODE:-packed}"
  for b in 8 64 1; do r $B --rollout $GSM --nseq 128 --bs $b --check --ref $REF/ref_gsm.pt --tag $TAG-gsm-b$b $M; done
  r $B --rollout $GSM --nseq 128 --bs 8 --order shuffle --check --ref $REF/ref_gsm.pt --tag $TAG-gsm-b8-shuffle $M
  for b in 4 1 8; do r $B --rollout $L4K --bs $b --check --ref $REF/ref_4k.pt --tag $TAG-4k-b$b $M; done
  r $B --rollout $L4K --bs 4 --order shuffle --check --ref $REF/ref_4k.pt --tag $TAG-4k-b4-shuffle $M ;;
stime) export CUDA_VISIBLE_DEVICES=$1; J=$O/$TAG.stime.jsonl; : > $J; M="--mode ${SMODE:-packed} ${SARGS:-}"
  r $B --rollout $GSM --bs 8 --reps 3 --tag $TAG-gsm-b8 $M
  for b in 1 4; do r $B --rollout $L4K --bs $b --reps 3 --tag $TAG-4k-b$b $M; done ;;
gate) export BITEXACT_VERIFY_HOST_ROWS=1; WL="--data $DATA --n-prompts 32 --g 8"; j=0
  for cfg in "t1 1.0 1" "t1 1.0 64" "t0 0 1" "t0 0 64"; do read n t ms <<< "$cfg"; j=$((j+1)); g=${!j}; out=$O/${TAG}_${n}_ms$ms.json
    CUDA_VISIBLE_DEVICES=$g $PY $X/sample.py --engine bitexact --model $INT $WL --temp $t --max-seqs $ms --out $out > $out.log 2>&1 &
  done; wait; grep -h RESULT $O/${TAG}_t*_ms*.json.log ;;
dspeed) export CUDA_VISIBLE_DEVICES=$1
  for bg in "1 512" "64 512" "512 256"; do read b n <<< "$bg"
    $PY $X/speed.py --model $INT --data $DATA --B $b --gen $n --out $O/${TAG}_speed_B$b.json > $O/${TAG}_speed_B$b.log 2>&1
    grep -h SPEED $O/${TAG}_speed_B$b.log; done ;;
bench) export CUDA_VISIBLE_DEVICES=$1; J=$O/${TAG}_bench.log; : > $J
  for rep in 1 2 3; do for b in 1 32 128; do for t in 0 1; do
    timeout 3000 $PY tests/vllm/bench.py --model $INT --batch $b --temp $t --tag $TAG-r$rep 2>&1 | grep '^RESULT' | tee -a $J
  done; done; done ;;
esac
