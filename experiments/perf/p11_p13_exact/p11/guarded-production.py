import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/data41/audreymao/perf11_20261105/gemm')
out=root/'out/guardedtokens8';out.mkdir(exist_ok=False)
trees={'base':Path('/data41/audreymao/perf10_20261105/exact_gemm/repo'),'repo':root/'guardrepo'}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
records=[]
for mode,options in (('mixed',[]),('greedy',['--temps','0']),('logprobs',['--logprobs'])):
    for name,tree in trees.items():
        tag=name+'_'+mode
        while True:
            m=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
            ps=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if m=='0' and not ps:break
            time.sleep(5)
        print('START',tag,flush=True)
        cmd=[sys.executable,'experiments/perf/p4_merge/tokens.py','--model','/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel','--out',str(out/(tag+'.json'))]+options
        with (out/(tag+'.log')).open('w') as log:
            p=subprocess.run(cmd,cwd=tree,env=env|{'PYTHONPATH':str(tree/'src')},stdout=log,stderr=subprocess.STDOUT,timeout=900)
        records.append({'tag':tag,'exit':p.returncode,'args':cmd});(out/'result.json').write_text(json.dumps(records,indent=2))
        print('END',tag,p.returncode,flush=True)
        assert p.returncode==0,tag
    a=json.loads((out/('base_'+mode+'.json')).read_text());b=json.loads((out/('repo_'+mode+'.json')).read_text())
    assert len(a)==len(b)==128 and all(len(t)==48 for t in a+b)
    assert a==b,mode
    print('MATCH',mode,128*48,flush=True)
