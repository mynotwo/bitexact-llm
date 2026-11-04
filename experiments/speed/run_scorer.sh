#!/usr/bin/env bash
# usage: run_item.sh check|time TAG GPU [extra bench.py args]   (extra args select the engine variant)
CODE=${CODE:-$2}; source $(dirname $0)/env_scorer.sh
kind=$1; tag=$2; G=$3; shift 3; X="$*"
export CUDA_VISIBLE_DEVICES=$G; cd $S
echo "== $kind $tag GPU $G $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader) $(date +%T)"
J=$O/$tag.$kind.jsonl; : > $J
r() { "$@" 2>&1 | grep '^{' | tee -a $J || true; }
if [ $kind = check ]; then
  for b in 8 64 1; do r $B --rollout $GSM --nseq 128 --bs $b --check --ref $O/ref_gsm.pt --tag $tag-gsm-b$b $X; done
  r $B --rollout $GSM --nseq 128 --bs 8 --order shuffle --check --ref $O/ref_gsm.pt --tag $tag-gsm-b8-shuffle $X
  for b in 4 1 8; do r $B --rollout $L4K --bs $b --check --ref $O/ref_4k.pt --tag $tag-4k-b$b $X; done
  r $B --rollout $L4K --bs 4 --order shuffle --check --ref $O/ref_4k.pt --tag $tag-4k-b4-shuffle $X
else
  r $B --rollout $GSM --bs 8 --reps 3 --tag $tag-gsm-b8 $X
  for b in 1 4; do r $B --rollout $L4K --bs $b --reps 3 --tag $tag-4k-b$b $X; done
fi
echo "DONE $kind $tag $(date +%T)"
