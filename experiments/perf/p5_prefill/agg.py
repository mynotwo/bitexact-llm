"""Aggregate RESULT lines: python agg.py DIR  -> median [min-max] per tag group (strip _rN)."""
import collections
import glob
import json
import os
import re
import statistics
import sys

d = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.log"))):
    name = os.path.basename(f)[:-4]
    for line in open(f):
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            v = r.get("tok_per_s_median", r.get("tok_per_s"))
            d[re.sub(r"_r\d+$", "", name)].append(v)
for k, v in sorted(d.items()):
    print(f"{k}: median {statistics.median(v):,.0f} [{min(v):,.0f}-{max(v):,.0f}] n={len(v)}")
