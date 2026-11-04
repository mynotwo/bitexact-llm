# profile env on 138 (H100): reuses the abc_measure venv, constants, tokenizer dir and GSM8K data read-only.
W=$HOME/abc_measure; P=$HOME/profile; O=$P/out; X=$P/code/experiments/profile
PY=$W/venv/bin/python; CONST=$W/const.pt; DEQ=$W/deqmodel; INT=$HOME/portability/intmodel; DATA=$W/gsm8k
export PYTHONPATH=$P/code/src TMPDIR=$P/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME=$P/cache TRITON_CACHE_DIR=$P/cache/triton VLLM_CACHE_ROOT=$P/cache/vllm \
  TORCHINDUCTOR_CACHE_DIR=$P/cache/inductor
mkdir -p $O $P/tmp $P/cache
freegpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | tr '\n' ' '; }
