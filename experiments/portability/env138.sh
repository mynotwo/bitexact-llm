# portability env on 138 (H100). Uses the existing abc_measure venv read-only (vllm 0.32.0, torch 2.13.0); code via PYTHONPATH.
W=$HOME/abc_measure; P=$HOME/portability; O=$P/out; X=$P/code/experiments/portability; R=$P/code/experiments/rl_mismatch
PY=$W/venv/bin/python; DEQ=$W/deqmodel; INT=$P/intmodel; CONST=$W/const.pt; DATA=$W/gsm8k
export PYTHONPATH=$P/code/src TMPDIR=$P/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME=$P/cache TRITON_CACHE_DIR=$P/cache/triton VLLM_CACHE_ROOT=$P/cache/vllm \
  TORCHINDUCTOR_CACHE_DIR=$P/cache/inductor BITEXACT_INT8_CONST=$CONST
mkdir -p $O $P/tmp $P/cache
# free GPUs = 0 MiB used
freegpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | tr '\n' ' '; }
