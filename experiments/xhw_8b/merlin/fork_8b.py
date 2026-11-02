"""Fork the L20 job ef1b5b5295886b0e (its image) with the 8B entry script, on a given GPU / queue.

usage: fork_8b.py NAME ENTRY.sh [CLUSTER_ID GPU_TYPE QUEUE_SHORT CPU MEM_MB]
defaults: 54 NVIDIA_L20 hl 22 117760 (the L20 hl spot run of the 1B check)
prints: NAME SID <sid> ok   (or NAME SID NONE <error text>)"""
import json
import os
import subprocess
import sys

M = os.environ.get("MERLIN_CLI", os.path.expanduser("~/.merlin-cli/bin/merlin-cli"))
SRC = "ef1b5b5295886b0e"
a = sys.argv[1:]
name, entry = a[0], a[1]
cluster, gpu_type, q, cpu, mem = (a[2:] + ["54", "NVIDIA_L20", "hl", "22", "117760"][len(a) - 2:])[:5]
queue = f"compute-598-{q}-gpuspot1-default-mlsys.public.inc-guarantee"
raw = subprocess.run([M, "job-v2", "runs", "get-request-config", "--json", json.dumps({"sid": SRC})],
                     capture_output=True, text=True).stdout
jt = json.loads(raw[raw.find("{"):], strict=False)["job_config"]["job_template_config"]
jt["entrypoint_full_script"] = open(entry).read()
rc = {"arnold_resource_config": {"cluster_ids": [int(cluster)], "cluster_select_mode": "single", "group_id": 271,
      "roles": [{"cpu": int(cpu), "gpu": 1, "gpu_type": gpu_type, "memory_mb": int(mem), "name": "worker", "num": 1,
                 "queue_name": queue}]}, "resource_backend": "arnold"}
req = {"source_sid": SRC, "name": name, "overrides": {"job_config": {"job_template_config": jt}, "resource_config": rc}}
out = subprocess.run([M, "job-v2", "runs", "fork", "--json", json.dumps(req)], capture_output=True, text=True)
txt = out.stdout + out.stderr
sid = [ln.split('"sid": "')[1].split('"')[0] for ln in txt.splitlines() if '"sid": "' in ln]
print(name, "SID", sid[-1] if sid else "NONE", "ok" if '"code": 0' in txt else txt[:600].replace("\n", " "))
