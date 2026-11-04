"""Print selected keys of the JSON lines on stdin (keeps lines short)."""
import json
import sys

KEYS = ("tag", "rows", "rows_equal_plugin_sha", "seqs", "seqs_hash_equal_ref", "seqs_logp_bits_equal_ref", "bit_exact",
        "tok_per_s", "sec")
for line in sys.stdin:
    if line.startswith("{"):
        d = json.loads(line)
        print(" ".join(f"{k}={d[k]}" for k in KEYS if k in d))
