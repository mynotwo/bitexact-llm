import os,subprocess,sys,time
from pathlib import Path
root=Path('/data41/audreymao/perf11_20261105/gemm')
while True:
    m=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
    ps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if m=='0' and not ps:break
    time.sleep(5)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(root/'guardrepo/src'))
cmd=[sys.executable,'experiments/perf/p10_exact_gemm/check_configs.py','/data41/audreymao/xhw8b/out/const_8b_public.pt',
    '--base','/data41/audreymao/perf10_20261105/exact_gemm/repo/src/bitexact/vllm_plugin/exact_gemm.py',
    '--rows','1,3,8,9,16,17,31,32,33,64,65,96,127,128,129,256,512',
    '--out',str(root/'out/guardedholdout8.json')]
print('START healthy GPU3 holdout',flush=True)
proc=subprocess.run(cmd,cwd=root/'guardrepo',env=env)
print('END healthy GPU3 holdout',proc.returncode,flush=True)
raise SystemExit(proc.returncode)
