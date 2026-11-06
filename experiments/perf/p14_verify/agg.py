"""Aggregate p14 bench RESULT lines: median [min, max] tok/s per (model, mode, batch, temp, tree).
Usage: python3 agg.py DIR_WITH_BENCH_LOGS   (reads *.log written by run.sh cell())."""
import glob, json, os, statistics as st, sys
d = sys.argv[1]
cells = {}
for f in glob.glob(os.path.join(d, '*.log')):
    for line in open(f):
        if line.startswith('RESULT '):
            j = json.loads(line[7:])
            tag = j['tag']  # e.g. e1_B128_T0_base_r3
            key = tag.rsplit('_r', 1)[0]
            cells.setdefault(key, []).append(j['tok_per_s'])
out = {}
for k in sorted(cells):
    v = cells[k]
    out[k] = {'n': len(v), 'median': st.median(v), 'min': min(v), 'max': max(v), 'values': v}
    print('%-22s n=%d median=%9.1f [%9.1f, %9.1f]' % (k, len(v), st.median(v), min(v), max(v)))
json.dump(out, open(os.path.join(d, 'bench_summary.json'), 'w'), indent=1)
