import json,os,subprocess,sys
from pathlib import Path
r=Path('/data41/audreymao/perf13_20261105/ew')
p=r/'out/fuse8';p.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(r/'repo/src'),BITEXACT_EXACT_CONST='/data41/audreymao/xhw8b/out/const_8b_public.pt')
records=[]
for name,cmd in [('kernels',[sys.executable,'-m','pytest','-q','tests/test_vllm_exact_fuse.py']),('edges',[sys.executable,'experiments/perf/p10_exact_gemm/p13_recip_edges.py'])]:
 assert subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()=='0'
 assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 print('START',name,flush=True)
 with (p/(name+'.log')).open('w') as log: proc=subprocess.run(cmd,cwd=r/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'args':cmd,'exit':proc.returncode});(p/'result.json').write_text(json.dumps(records,indent=2))
 print('END',name,proc.returncode,flush=True)
 assert proc.returncode==0,name
