# speed_decode env on 138 (H100): abc_measure venv read-only; code via PYTHONPATH from ~/speed_decode/code (rsync of the worktree).
W=$HOME/abc_measure; P=$HOME/speed2d; O=$P/out; X=$P/code/experiments/speed_decode
PY=$W/venv/bin/python; DEQ=$W/deqmodel; INT=$HOME/portability/intmodel; CONST=$W/const.pt; DATA=$W/gsm8k
export PYTHONPATH=$P/code/src TMPDIR=$P/tmp HF_HOME=$W/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME=$P/cache TRITON_CACHE_DIR=$P/cache/triton VLLM_CACHE_ROOT=$P/cache/vllm \
  TORCHINDUCTOR_CACHE_DIR=$P/cache/inductor BITEXACT_INT8_CONST=$CONST
mkdir -p $O $P/tmp $P/cache
freegpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1}' | tr '\n' ' '; }
