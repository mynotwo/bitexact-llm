"""Run isolated exact-mode checks and measurements on the existing H800 host."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data41/audreymao/perf10_20261105/exact_gemm')
MODELS = Path('/data41/audreymao/perf7_20261105/exactplugin')


def source_hash(tree):
    digest = hashlib.sha256()
    for path in sorted((tree / 'src').rglob('*.py')):
        digest.update(str(path.relative_to(tree)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def wait_gpu(gpu):
    assert gpu in (0, 1, 2, 3), 'GPU outside allocation'
    deadline = time.monotonic() + 3600
    while True:
        common = ['nvidia-smi', '-i', str(gpu), '--format=csv,noheader,nounits']
        memory = subprocess.check_output(common + ['--query-gpu=memory.used'], text=True).strip()
        processes = subprocess.check_output(common + ['--query-compute-apps=pid'], text=True).strip()
        if memory == '0' and not processes:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'GPU {gpu} remained busy for one hour')
        print(f'waiting for GPU {gpu}: memory={memory}, pids={processes}', flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('sweep', 'suite', 'pytest', 'kernels', 'tokens', 'bench',
                                         'profile', 'check8', 'configs'))
    parser.add_argument('--tree', choices=('base', 'repo'), default='repo')
    parser.add_argument('--size', type=int, choices=(1, 8), default=1)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--rows', default='1,32,128')
    parser.add_argument('--names', default='wqkv,wo,wgu,wd,lm')
    parser.add_argument('--reps', type=int, default=5)
    args = parser.parse_args()
    output = ROOT / 'out' / args.tag
    output.mkdir(parents=True, exist_ok=False)
    model = MODELS / ('exactmodel' if args.size == 1 else 'exact8bmodel')
    const = json.loads((model / 'config.json').read_text())['bitexact_const']
    assert Path(const).is_file()
    tree = ROOT / args.tree
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), GPUS=str(args.gpu),
                       VLLM_ENABLE_V1_MULTIPROCESSING='0', PY=sys.executable, MODEL=str(model),
                       CONST=const, OUT=str(output), BITEXACT_EXACT_CONST=const,
                       BITEXACT_INT8_CONST='/data41/audreymao/a17_vllm/data/intllama_const.pt',
                       BITEXACT_INT8_CONST_8B='/data41/audreymao/p3_8b/int8_const_8b.pt',
                       BITEXACT_VLLM_MODEL='/data41/audreymao/perf_20261030/det_compare/intmodel',
                       BITEXACT_MODEL='/data41/audreymao/perf_20261030/bf16model')
    result = {'args': vars(args), 'constant': const, 'model': str(model), 'source_sha256': source_hash(tree),
              'commands': [], 'started': time.time(), 'status': 'RUNNING'}

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(name, command, timeout=10800, selected_tree=tree):
        wait_gpu(args.gpu)
        env = environment | {'PYTHONPATH': f'{selected_tree}/src:{selected_tree}/tests/vllm'}
        started = time.time()
        print(f'START {name} gpu={args.gpu}', flush=True)
        with (output / f'{name}.log').open('x') as log:
            proc = subprocess.run(command, cwd=selected_tree, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout, check=False)
        result['commands'].append({'name': name, 'argv': command, 'exit_code': proc.returncode,
                                   'seconds': time.time() - started, 'tree': selected_tree.name,
                                   'source_sha256': source_hash(selected_tree)})
        save()
        print(f'END {name} rc={proc.returncode}', flush=True)
        if proc.returncode:
            raise RuntimeError(f'{name} failed: inspect its log in {output}')

    save()
    try:
        if args.stage == 'sweep':
            script = ROOT / 'repo/experiments/perf/p10_exact_gemm/sweep.py'
            run('sweep', [sys.executable, str(script), const, '--rows', args.rows, '--names', args.names,
                          '--out', str(output / 'sweep.json')], timeout=21600)
        elif args.stage == 'configs':
            script = ROOT / 'repo/experiments/perf/p10_exact_gemm/check_configs.py'
            base = ROOT / 'base/src/bitexact/vllm_plugin/exact_gemm.py'
            run('configs', [sys.executable, str(script), const, '--base', str(base),
                            '--out', str(output / 'configs.json')], timeout=21600)
        elif args.stage == 'suite':
            run('suite', ['bash', 'tests/vllm/run_tests.sh', args.tree], timeout=36000)
        elif args.stage in ('pytest', 'kernels'):
            tests = 'tests/' if args.stage == 'pytest' else 'tests/test_vllm_exact_gemm.py'
            run(args.stage, [sys.executable, '-m', 'pytest', '-q', tests])
        elif args.stage == 'check8':
            assert args.size == 8
            prefix = str(output / 'reference')
            run('reference', [sys.executable, 'experiments/perf/p7_exactplugin/ref8.py',
                              '--const', const, '--out', prefix])
            for eager in (0, 1):
                name = f'batch_eager{eager}'
                path = str(output / f'{name}.json')
                run(name, [sys.executable, 'experiments/perf/p7_exactplugin/batch_inv.py',
                           '--model', str(model), '--out', path, '--eager', str(eager)])
                run(f'compare_eager{eager}', [sys.executable, 'experiments/perf/p7_exactplugin/cmp8.py',
                                             prefix + '.hash', path + '.hash'])
        elif args.stage == 'tokens':
            for name, options in (('mixed', []), ('greedy', ['--temps', '0']), ('logprobs', ['--logprobs'])):
                run(name, [sys.executable, 'experiments/perf/p4_merge/tokens.py', '--model', str(model),
                           '--out', str(output / f'{name}.json'), *options])
        elif args.stage == 'bench':
            for rep in range(args.reps):
                for batch in (1, 32, 128):
                    for temp in ((0, 1) if args.size == 1 else (0,)):
                        order = ('base', 'repo') if rep % 2 == 0 else ('repo', 'base')
                        for item in order:
                            tag = f'{item}_b{batch}_t{temp}_r{rep}'
                            run(tag, [sys.executable, 'tests/vllm/bench.py', '--model', str(model),
                                      '--batch', str(batch), '--temp', str(temp), '--tag', tag],
                                selected_tree=ROOT / item)
        else:
            for item in ('base', 'repo'):
                run(f'profile_{item}', [sys.executable, 'experiments/perf/p8_exactfuse/prof_vllm.py',
                                      '--model', str(model), '--batch', '128', '--temp', '0', '--tag', item,
                                      '--out', str(output / item)], selected_tree=ROOT / item)
        result['status'] = 'PASS'
    except Exception as exc:
        result['status'] = 'FAIL'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
