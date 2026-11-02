"""Extract the JSONs a Merlin job printed (RESULT_B64_BEGIN <name> ... RESULT_B64_END) from its stdout log.

usage: extract.py JOB.log OUTDIR"""
import base64
import gzip
import os
import sys

log, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
name, buf = None, []
for line in open(log, errors="replace"):
    line = line.strip()
    if line.startswith("RESULT_B64_BEGIN"):
        name, buf = line.split()[1], []
    elif line == "RESULT_B64_END" and name:
        path = os.path.join(out, name + ".json")
        open(path, "wb").write(gzip.decompress(base64.b64decode("".join(buf))))
        print("wrote", path)
        name = None
    elif name:
        buf.append(line)
