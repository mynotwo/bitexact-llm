#!/usr/bin/env bash
# Prefix-caching tax: the same cell with enable_prefix_caching=1 and =0, alternating, two reps each.
# bench_pc.py is tests/vllm/bench.py with `enable_prefix_caching=(PC == "1")`; the profiler forces it off, so
# this is the only way to quote a profiler number next to a bench number.
#   bash pcbench.sh start   (syncs to 141 and runs, 1B cells on GPU 2, 8B on GPU 3)
#   bash pcbench.sh agg
set -u
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf9_20261105/merge
A=/data41/audreymao/a17_vllm
D=/data41/audreymao/perf_20261030
P=/data41/audreymao/perf7_20261105/exactplugin
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
case "${1:-start}" in
sync) ssh $HOST "mkdir -p $S/pc"; scp -q "$LOCAL/experiments/perf/p9_merge/pcbench.sh" "$LOCAL/experiments/perf/p9_merge/bench_pc.py" $HOST:$S/pc/; exit ;;
agg)
  T=$(mktemp -d); ssh $HOST "cd $S/pc && tar cf - *.log" | tar xf - -C "$T"
  python3 - "$T" <<'PY'
import glob, json, os, statistics as st, sys
d = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*_pc*_r*.log'))):
    for l in open(f):
        if l.startswith('RESULT '):
            j = json.loads(l[7:]); cell, pc, rep = j['tag'].rsplit('_', 2)
            d.setdefault((cell, pc), []).append(j['tok_per_s'])
for k, v in sorted(d.items()):
    print('%-9s %-3s %s' % (k[0], k[1], [round(x) for x in v]))
print()
for cell in sorted({c for c, _ in d}):
    on, off = d.get((cell, 'pc1'), []), d.get((cell, 'pc0'), [])
    if on and off:
        a, b = st.median(on), st.median(off)
        print('%-9s pc1 %8.1f  pc0 %8.1f  %4.1f%% lost' % (cell, a, b, 100 * (a - b) / a))
PY
  rm -rf "$T"; exit ;;
esac

. $A/env.sh
PY=$A/venv/bin/python
R=$S/repo
export VLLM_ENABLE_V1_MULTIPROCESSING=0
O=$S/pc; mkdir -p $O
E1=$P/exactmodel E8=$P/exact8bmodel
I1=$D/det_compare/intmodel I8=/data41/audreymao/p3_8b/intmodel8b
B1=$D/bf16model B8=/data08/models/hf-cache/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
wait_gpu() { while ! free $1; do sleep 30; done; sleep 3; }
cell() { gpu=$1 model=$2 name=$3 batch=$4
  case " 0 1 2 3 7 " in *" $gpu "*) ;; *) echo "GPU $gpu not assigned"; exit 2;; esac
  for rep in 1 2; do for pc in 1 0; do
    wait_gpu $gpu
    ( cd $R && PC=$pc CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R/src:$R/tests/vllm timeout 2400 \
      $PY $S/pc/bench_pc.py --model $model --batch $batch --tag ${name}_pc${pc}_r${rep} ) > $O/${name}_pc${pc}_r${rep}.log 2>&1
    echo "$(date +%T) ${name} pc${pc} r${rep} $(grep -h '^RESULT' $O/${name}_pc${pc}_r${rep}.log | grep -o '"tok_per_s": [0-9.]*')"
  done; done; }
oneb() { cell 2 $E1 1b_exact 128; cell 2 $I1 1b_int8 128; cell 2 $B1 1b_bf16 128; }
eightb() { cell 3 $E8 8b_exact 128; cell 3 $I8 8b_int8 128; }
oneb & eightb &
wait
echo "$(date +%T) PCDONE" | tee -a $O/SUMMARY.txt