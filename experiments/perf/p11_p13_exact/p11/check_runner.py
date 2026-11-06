import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/data41/audreymao/perf11_20261105/gemm')
base=Path('/data41/audreymao/perf10_20261105/exact_gemm/repo')
mode=sys.argv[1];gpu=int(sys.argv[2]);size=int(sys.argv[3])
assert gpu in (0,1,2,3)
tag=mode+str(size);out=root/'out'/tag;out.mkdir(exist_ok=False)
const='/data41/audreymao/xhw8b/out/const_8b_public.pt' if size==8 else '/data41/audreymao/perf_20261030/exact_tput/consts/exact.pt'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root/'repo/src'),BITEXACT_EXACT_CONST=const)
commands=[]
def run(name,args):
    assert subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()=='0'
    assert not subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    print('START',name,flush=True)
    with (out/(name+'.log')).open('w') as f:
        p=subprocess.run(args,cwd=root/'repo',env=env,stdout=f,stderr=subprocess.STDOUT)
    commands.append({'name':name,'args':args,'exit':p.returncode})
    (out/'result.json').write_text(json.dumps(commands,indent=2))
    print('END',name,p.returncode,flush=True)
    assert p.returncode==0,(name,p.returncode)
if mode=='correct':
    run('kernels',[sys.executable,'-m','pytest','-q','tests/test_vllm_exact_gemm.py'])
    run('boundaries',[sys.executable,'experiments/perf/p10_exact_gemm/check_configs.py',const,
                      '--base',str(base/'src/bitexact/vllm_plugin/exact_gemm.py'),
                      '--rows','1,3,8,9,16,17,31,32,33,64,65,96,127,128,129,256,512',
                      '--out',str(out/'boundaries.json')])
else:
    model='/data41/audreymao/perf7_20261105/exactplugin/exact8bmodel'
    prefix=str(out/'ref')
    run('reference',[sys.executable,'experiments/perf/p7_exactplugin/ref8.py','--const',const,'--out',prefix])
    for eager in (0,1):
        path=str(out/f'batch{eager}.json')
        run(f'batch{eager}',[sys.executable,'experiments/perf/p7_exactplugin/batch_inv.py','--model',model,'--out',path,'--eager',str(eager)])
        run(f'compare{eager}',[sys.executable,'experiments/perf/p7_exactplugin/cmp8.py',prefix+'.hash',path+'.hash'])
