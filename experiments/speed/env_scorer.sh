# speed-scorer env on 138 (H100): reuses the abc_measure venv, constants, tokenizer dir, GSM8K data and the profile rollouts.
W=$HOME/abc_measure; S=$HOME/speed2; O=$S/out; P=$HOME/profile
PY=$W/venv/bin/python; CONST=$W/const.pt; DEQ=$W/deqmodel; DATA=$W/gsm8k
GSM=$P/roll/q1_B.json; L4K=$P/roll/q2_4096_B.json
export PYTHONPATH=$S/code_$CODE/src TMPDIR=$S/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$S/cache TRITON_CACHE_DIR=$S/cache/triton \
  TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor
mkdir -p $O $S/tmp $S/cache
freegpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | tr '\n' ' '; }
B="$PY $S/code_$CODE/experiments/speed_scorer/bench.py --const $CONST --model $DEQ --data $DATA"
