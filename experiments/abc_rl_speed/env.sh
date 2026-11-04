# env on 138 for the speed comparison: one venv (abc_measure: vLLM 0.32.0, torch 2.13.0+cu130) for all arms.
# usage: source env.sh TAG   (TAG = code dir suffix: code_$TAG holds src/ and experiments/)
S=$HOME/abc_rl_speed
export PY=$HOME/abc_measure/venv/bin/python
export HF=$(ls -d $HOME/genzip/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/*/ | head -1)
export CONST=$HOME/rl_mismatch/intllama_const.pt DATA=$HOME/rl_mismatch/data
export PYTHONPATH=$S/code_$1/src TMPDIR=$S/tmp HF_HOME=$HOME/abc_measure/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=$S/cache_$2 XDG_CACHE_HOME=$S/cache_$2 TRITON_CACHE_DIR=$S/cache_$2/triton \
  TORCHINDUCTOR_CACHE_DIR=$S/cache_$2/inductor VLLM_CACHE_ROOT=$S/cache_$2/vllm
export C=$S/code_$1/experiments/abc_rl
mkdir -p $S/tmp $S/cache_$2 $S/runs
