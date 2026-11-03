#!/usr/bin/env bash
# 8B plugin runs on one GPU, each stage waits until the GPU is empty (0 MiB, no compute process).
#   S=/data41/audreymao/p3_8b GPU=0 bash experiments/perf/p3_8b/run.sh STAGE...
# stages: model ref bi suite bench
set -u
S=${S:-/data41/audreymao/p3_8b}; GPU=${GPU:-0}; R=$S/repo; O=$S/out; mkdir -p $O
source /data41/audreymao/a17_vllm/env.sh
export PYTHONPATH=$R/src; PY=/data41/audreymao/a17_vllm/venv/bin/python
HF=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
C=$S/int8_const_8b.pt; M=$S/intmodel8b
free() { [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 400 ] && [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { while ! free; do sleep 15; done; sleep 5; free || wait_gpu; }
run() { name=$1; shift; wait_gpu; echo "$(date +%T) $name"; CUDA_VISIBLE_DEVICES=$GPU timeout 3000 env "$@" > $O/$name.log 2>&1; echo "  exit $?"; }
cd $R
for st in "$@"; do case $st in
 model) $PY -m bitexact.vllm_plugin.make_model_dir --hf $HF --const $C --out $M ;;
 ref) run ref $PY tests/vllm/run_ref.py --const $C --out $O/ref ;;
 bi) run bi_graph $PY experiments/perf/p3_8b/batch_inv.py --model $M --out $O/bi_graph.json --eager 0
     run bi_eager env BITEXACT_GRAPHS=0 $PY experiments/perf/p3_8b/batch_inv.py --model $M --out $O/bi_eager.json --eager 1 ;;
 suite) RS="$PY tests/vllm/run_serve.py --model $M"
     run alone_pc1 $RS --mode alone --prefix-cache 1 --out $O/alone_pc1
     run busy_pc1_s0 $RS --mode busy --prefix-cache 1 --arrival-seed 0 --out $O/busy_pc1_s0
     run busy_pc0_s1 $RS --mode busy --prefix-cache 0 --arrival-seed 1 --out $O/busy_pc0_s1
     run busy_pc1_s2_b64 $RS --mode busy --prefix-cache 1 --arrival-seed 2 --budget 64 --out $O/busy_pc1_s2_b64
     run busy_eager_s4 env BITEXACT_GRAPHS=0 BITEXACT_COMPILE=0 $RS --mode busy --prefix-cache 1 --arrival-seed 4 --max-seqs 128 --out $O/busy_eager_s4
     run busy_vllmgraph_s5 $RS --mode busy --prefix-cache 1 --arrival-seed 5 --eager 0 --out $O/busy_vllmgraph_s5
     run alone_vllmgraph $RS --mode alone --prefix-cache 1 --eager 0 --out $O/alone_vllmgraph
     $PY tests/vllm/compare.py $O/ref $O/alone_pc1 $O/busy_pc1_s0 $O/busy_pc0_s1 $O/busy_pc1_s2_b64 $O/busy_eager_s4 $O/busy_vllmgraph_s5 $O/alone_vllmgraph > $O/compare.txt 2>&1; echo compare_exit $? ;;
 rest) export GPU_UTIL=0.6; RS="$PY tests/vllm/run_serve.py --model $M"
     run ref $PY tests/vllm/run_ref.py --const $C --out $O/ref
     run busy_eager_s4 env BITEXACT_GRAPHS=0 BITEXACT_COMPILE=0 $RS --mode busy --prefix-cache 1 --arrival-seed 4 --max-seqs 128 --out $O/busy_eager_s4
     run alone_vllmgraph $RS --mode alone --prefix-cache 1 --eager 0 --out $O/alone_vllmgraph
     $PY tests/vllm/compare.py $O/ref $O/alone_pc1 $O/busy_pc1_s0 $O/busy_pc0_s1 $O/busy_pc1_s2_b64 $O/busy_eager_s4 $O/busy_vllmgraph_s5 $O/alone_vllmgraph > $O/compare.txt 2>&1; echo compare_exit $? ;;
 bench) for b in 1 32 128; do for rep in 1 2 3; do
     run int_b${b}_r$rep $PY tests/vllm/bench.py --model $M --batch $b
     run bf16_b${b}_r$rep $PY tests/vllm/bench.py --model $HF --batch $b
     run bi_b${b}_r$rep env VLLM_BATCH_INVARIANT=1 $PY tests/vllm/bench.py --model $HF --batch $b
   done; done; grep -h "^RESULT" $O/*_b*_r*.log ;;
esac; done
