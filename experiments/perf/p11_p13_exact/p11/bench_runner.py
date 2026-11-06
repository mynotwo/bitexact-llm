import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/data41/audreymao/perf11_20261105/gemm')
trees={'base':Path('/data41/audreymao/perf10_20261105/exact_gemm/repo'),'repo':root/'repo'}
out=root/'out/bench8';out.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',VLLM_ENABLE_V1_MULTIPROCESSING='0')
model='/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel'
def source(tree):
    h=hashlib.sha256()
    for p in sorted((tree/'src').rglob('*.py')):
        if p.name.startswith('._'):continue
        h.update(str(p.relative_to(tree)).encode()+b'\0'+p.read_bytes())
    return h.hexdigest()
results={'source_sha256':{k:source(t) for k,t in trees.items()},'commands':[],'status':'RUNNING'}
def save(): (out/'result.json').write_text(json.dumps(results,indent=2))
save()
try:
    for rep in range(5):
        for batch in (32,128,1):
            for name in (('base','repo') if rep%2==0 else ('repo','base')):
                tag=f'{name}_b{batch}_r{rep}'
                assert subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()=='0'
                assert not subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                print('START',tag,flush=True)
                cmd=[sys.executable,'tests/vllm/bench.py','--model',model,'--batch',str(batch),'--temp','0','--tag',tag]
                with (out/(tag+'.log')).open('w') as log:
                    p=subprocess.run(cmd,cwd=trees[name],env=env|{'PYTHONPATH':str(trees[name]/'src')},stdout=log,stderr=subprocess.STDOUT)
                assert source(trees[name])==results['source_sha256'][name],'source changed during bench'
                results['commands'].append({'tag':tag,'exit':p.returncode,'args':cmd,'source_sha256':source(trees[name])});save()
                print('END',tag,p.returncode,flush=True)
                assert p.returncode==0,tag
    results['status']='PASS'
except Exception as e:
    results['status']='FAIL';results['error']=str(e);raise
finally:save()
