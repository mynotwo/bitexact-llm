"""Validate raw GPU evidence for the exact GEMM change; no CUDA needed locally."""
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / 'out'
BASE = '3bcaf9028e549d659c8a560186e5587e7af6b3ef196fed89216c8af4234fca13'
ROWS = (1, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 127, 128, 129, 256, 512)
NAMES = ('wqkv', 'wo', 'wgu', 'wd', 'lm')
SHORT = ('ref', 'alone_pc1', 'alone_pc0', 'busy_pc1_s0', 'busy_pc0_s1',
         'busy_pc1_s2_b64', 'busy_pc1_s3_preempt', 'busy_eager_s4',
         'busy_vllmgraph_s5', 'alone_vllmgraph')
LONG = ('long_ref', 'long_alone', 'long_busy', 'long_busy_eager')


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read(path):
    require(path.is_file() and path.stat().st_size > 0, f'missing/empty {path}')
    return path.read_text()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f'duplicate JSON key {key}')
        result[key] = value
    return result


def data(path):
    return json.loads(read(path), object_pairs_hook=pairs)


def source_hash():
    digest = hashlib.sha256()
    contents = {}
    for path in sorted((REPO / 'src').rglob('*.py')):
        contents[str(path.relative_to(REPO))] = path.read_bytes()
    # The macOS export added one non-importable AppleDouble sidecar. Bind its
    # exact bytes as the remote runner did, without dropping any source check.
    sidecar = (OUT / 'exact_gemm.appledouble').read_bytes()
    require(sidecar[:4] == bytes.fromhex('00051607'), 'invalid AppleDouble sidecar')
    sidecar_path = 'src/bitexact/vllm_plugin/._exact_gemm.py'
    exported_sidecar = contents.pop(sidecar_path, None)
    require(exported_sidecar is None or exported_sidecar == sidecar, 'export metadata changed')
    inventory = data(OUT / 'source_inventory.json')
    actual = {p: hashlib.sha256(b).hexdigest() for p, b in contents.items()}
    require(actual == inventory['local'], 'local source changed after inventory')
    require(all(hashlib.sha256((REPO / p).read_bytes()).hexdigest() == h
                for p, h in inventory['support'].items()), 'test/runner changed after remote inventory')
    contents[sidecar_path] = sidecar
    require({p: hashlib.sha256(b).hexdigest() for p, b in contents.items()} == inventory['remote'],
            'remote export differs from local source plus recorded metadata')
    for path, content in sorted(contents.items()):
        digest.update(path.encode() + b'\0' + content)
    return digest.hexdigest()


def manifest(tag, stage, size, gpu, commands, tree='repo'):
    folder = OUT / tag
    result = data(folder / 'result.json')
    require(result['status'] == 'PASS', f'{tag}: unfinished/failed')
    args = result['args']
    require((args['stage'], args['tree'], args['size'], args['gpu'], args['tag']) ==
            (stage, tree, size, gpu, tag), f'{tag}: incorrect invocation')
    require(result['source_sha256'] == (BASE if tree == 'base' else source_hash()), f'{tag}: stale source')
    model_name = 'exactmodel' if size == 1 else 'exact8bmodel'
    environment = data(OUT / 'environment.json')
    require(Path(result['model']).name == model_name and
            result['constant'] == environment['models'][model_name]['constant'], f'{tag}: wrong model/constants')
    require([c['name'] for c in result['commands']] == commands, f'{tag}: missing/reordered commands')
    for command in result['commands']:
        require(command['exit_code'] == 0, f'{tag}: {command["name"]} failed')
        selected = command['tree']
        require(selected in ('base', 'repo'), f'{tag}: unknown command tree')
        if stage == 'bench':
            require(command['name'].startswith(selected + '_'), f'{tag}: swapped benchmark labels')
        elif stage == 'profile':
            require(command['name'] == 'profile_' + selected, f'{tag}: swapped profile labels')
        else:
            require(selected == tree, f'{tag}: executed the wrong tree')
        expected = BASE if command['tree'] == 'base' else source_hash()
        require(command['source_sha256'] == expected, f'{tag}: stale command source')
        log = read(folder / (command['name'] + '.log'))
        require('Traceback (most recent call last)' not in log, f'{tag}: traceback')
    return folder


def hashes(path, seeds, length):
    found = {}
    for line in read(path).splitlines():
        parts = line.split()
        require(len(parts) == 4, f'{path}: malformed hash row')
        seed, pos, token = int(parts[0]), int(parts[1]), int(parts[3])
        require(re.fullmatch('[0-9a-f]{16}', parts[2]), f'{path}: invalid hash')
        if seed not in seeds:
            continue
        key, value = (seed, pos), (parts[2], token)
        require(key not in found or found[key] == value, f'{path}: inconsistent repeated row {key}')
        found[key] = value
    expected = {(s, t) for s in seeds for t in range(length)}
    require(set(found) == expected, f'{path}: missing/extra target positions')
    return found


def suite(tag, tree, gpu):
    folder = manifest(tag, 'suite', 1, gpu, ['suite'], tree)
    log = read(folder / 'suite.log')
    for marker in ('compare_exit 0', 'compare_long_exit 0'):
        require(marker in log.splitlines(), f'{tag}: missing successful {marker}')
    output = []
    for names, seeds, length in ((SHORT, set(range(1, 13)), 48), (LONG, {101, 102, 103}, 16)):
        reference = None
        for name in names:
            prefix = folder / f'{tree}_{name}'
            tokens = {int(k): v for k, v in data(prefix.with_suffix('.json')).items()}
            require(set(tokens) == seeds, f'{prefix}: wrong target seeds')
            require(all(len(v) == length for v in tokens.values()), f'{prefix}: truncated tokens')
            values = hashes(prefix.with_suffix('.hash'), seeds, length)
            require(all(values[s, t][1] == tokens[s][t] for s, t in values), f'{prefix}: hash/token mismatch')
            if reference is None:
                reference = tokens, values
            require((tokens, values) == reference, f'{prefix}: differs from standalone reference')
        output.append(reference)
    preempt = read(folder / f'{tree}_busy_pc1_s3_preempt.log')
    require(any(int(x) > 0 for x in re.findall(r'^preemptions (\d+)$', preempt, re.M)),
            f'{tag}: forced-preemption scenario was not exercised')
    return output


def check8(tag, tree, gpu):
    commands = ['reference', 'batch_eager0', 'compare_eager0', 'batch_eager1', 'compare_eager1']
    folder = manifest(tag, 'check8', 8, gpu, commands, tree)
    tokens = data(folder / 'reference.json')
    require(set(tokens) == {'0', '1', '2', '3'}, f'{tag}: invalid reference seeds')
    require(all(len(v) == 48 for v in tokens.values()), f'{tag}: short reference')
    reference = hashes(folder / 'reference.hash', set(range(4)), 48)
    require(all(reference[s, t][1] == tokens[str(s)][t] for s, t in reference), f'{tag}: reference tokens')
    for eager in (0, 1):
        result = data(folder / f'batch_eager{eager}.json')
        require(result['invariant_32_128'] is True and result['n_hash_rows'] == 192,
                f'{tag}: batch invariance failed')
        require(result['tokens'] == [tokens[str(s)] for s in range(4)], f'{tag}: token mismatch')
        observed = hashes(folder / f'batch_eager{eager}.json.hash', set(range(4)), 48)
        require(observed == reference, f'{tag}: logits mismatch')
        require('ref rows 192 plugin rows matched keys 192 diff 0' in
                read(folder / f'compare_eager{eager}.log'), f'{tag}: comparison absent')
    return tokens, reference


def configs(size, gpu):
    folder = manifest(f'final_configs{size}', 'configs', size, gpu, ['configs'])
    cells = data(folder / 'configs.json')
    require(len(cells) == 80 and {(c['name'], c['rows']) for c in cells} ==
            {(n, r) for n in NAMES for r in ROWS}, f'configs{size}: incomplete cells')
    for cell in cells:
        require(cell['equal_layers'] == (1 if cell['name'] == 'lm' else 5), 'missing layer comparisons')
        for key in ('base_us', 'repo_us'):
            require(len(cell[key]) == 3 and all(math.isfinite(t) and t > 0 for t in cell[key]), 'invalid timing')
        ratio = statistics.median(cell['base_us']) / statistics.median(cell['repo_us'])
        require(math.isclose(ratio, cell['ratio']), 'incorrect speed ratio')
    return cells


def benchmarks(size, gpu):
    expected = []
    temps = (0, 1) if size == 1 else (0,)
    for rep in range(5):
        for batch in (1, 32, 128):
            for temp in temps:
                for tree in (('base', 'repo') if rep % 2 == 0 else ('repo', 'base')):
                    expected.append(f'{tree}_b{batch}_t{temp}_r{rep}')
    folder = manifest(f'final_bench{size}', 'bench', size, gpu, expected)
    groups = {}
    for name in expected:
        records = [json.loads(line[7:]) for line in read(folder / f'{name}.log').splitlines()
                   if line.startswith('RESULT ')]
        require(len(records) == 1, f'{name}: missing/duplicate benchmark result')
        record = records[0]
        tree, b, t, _ = name.split('_')
        batch, temp = int(b[1:]), int(t[1:])
        require(record['tag'] == name and record['batch'] == batch and record['temp'] == temp,
                f'{name}: workload mismatch')
        require(not record['eager'] and record['in_len'] == record['out_len'] == 128 and
                record['out_tokens'] == batch * 128, f'{name}: shortened workload')
        require(record['sec'] > 0 and record['tok_per_s'] > 0 and
                math.isfinite(record['tok_per_s']), f'{name}: invalid timing')
        # bench.py rounds seconds to milliseconds and throughput to 0.1 tokens/s.
        require(abs(record['out_tokens'] / record['tok_per_s'] - record['sec']) < 0.002,
                f'{name}: inconsistent units')
        groups.setdefault((batch, temp), {'base': [], 'repo': []})[tree].append(record['tok_per_s'])
    summary = []
    for (batch, temp), pair in groups.items():
        require(all(len(v) == 5 for v in pair.values()), 'missing benchmark repetitions')
        medians = {k: statistics.median(v) for k, v in pair.items()}
        summary.append({'batch': batch, 'temp': temp, **pair, 'medians': medians,
                        'ratio': medians['repo'] / medians['base']})
    return summary


def tokens():
    commands = ['mixed', 'greedy', 'logprobs']
    old = manifest('base_tokens1', 'tokens', 1, 3, commands, 'base')
    new = manifest('final_tokens1', 'tokens', 1, 1, commands)
    for name in commands:
        a, b = data(old / f'{name}.json'), data(new / f'{name}.json')
        require(len(a) == len(b) == 128 and all(len(v) == 48 for v in a + b), 'truncated token streams')
        require(a == b, f'{name}: production tokens differ')


def tests():
    folder = manifest('final_pytest', 'pytest', 1, 1, ['pytest'])
    log = read(folder / 'pytest.log')
    require('147 passed, 8 skipped' in log and not re.search(r'\b\d+ (failed|error)\b', log),
            'pytest did not run the full 155-case collection (8 exact-only/int8 skips)')
    folder = manifest('final_kernels8', 'kernels', 8, 3, ['kernels'])
    require('1 passed, 1 skipped' in read(folder / 'kernels.log'), '8B exact GEMM test not executed')


def profiles(size, gpu):
    folder = manifest(f'final_profile{size}', 'profile', size, gpu, ['profile_base', 'profile_repo'])
    spec = importlib.util.spec_from_file_location('profile_buckets', HERE.parent / 'p8_exactfuse/buckets.py')
    buckets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(buckets)
    for tree in ('base', 'repo'):
        result = data(folder / f'{tree}.json')
        require(result['batch'] == 128 and result['out_tokens'] == 16384, 'profile workload mismatch')
        require(f'/{tree}/src/bitexact' == result['bitexact_src'][-len(f'/{tree}/src/bitexact'):],
                'profile imported wrong tree')
        require(result['timing']['steps'] == 128 and result['decode']['kernel_ms_per_step'] > 0,
                'empty profile')
        for phase in ('prefill', 'decode'):
            path = folder / f'{tree}.{phase}.trace.json.gz'
            require(path.is_file() and path.stat().st_size > 0, f'missing trace {path}')
            expected = buckets.summarize(str(path), 0 if phase == 'prefill' else 8,
                                         1 if phase == 'prefill' else 16)
            require(all(result[phase].get(k) == value for k, value in expected.items()),
                    f'{path}: stored profile differs from raw trace')


def main():
    require(suite('base_exact1_suite', 'base', 0) == suite('final_suite1', 'repo', 1),
            'baseline/candidate standalone 1B references differ')
    require(check8('base_exact8_check', 'base', 1) == check8('final_check8', 'repo', 3),
            'baseline/candidate standalone 8B references differ')
    tests()
    tokens()
    for size, gpu in ((1, 0), (8, 2)):
        cells = configs(size, gpu)
        print(f'{size}B: {len(cells)} exact configuration cells; output and flag assertions passed')
        for row in benchmarks(size, gpu):
            print(json.dumps({'model': size, **row}))
        profiles(size, gpu)
    print('PASS: source hashes, raw tokens/logits, boundary tests, complete paired timings and profiles')


if __name__ == '__main__':
    main()
