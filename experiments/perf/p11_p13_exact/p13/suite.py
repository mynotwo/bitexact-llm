import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/data41/audreymao/perf13_20261105/ew')
out=root/'out/fullsuite1';out.mkdir(exist_ok=False)
gpu=2
while True:
    memory=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if memory=='0' and not processes:break
    time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',GPUS='2',PY=sys.executable,OUT=str(out),
    PYTHONPATH=str(root/'repo/src')+':'+str(root/'repo/tests/vllm'),
    MODEL='/data41/audreymao/perf7_20261105/exactplugin/exactmodel',
    CONST='/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt',VLLM_ENABLE_V1_MULTIPROCESSING='0')
print('START suite',flush=True)
with (out/'suite.log').open('w') as log:
    proc=subprocess.run(['bash','tests/vllm/run_tests.sh','p13'],cwd=root/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
(out/'result.json').write_text(json.dumps({'exit':proc.returncode}))
print('END suite',proc.returncode,flush=True)
