#!/usr/bin/env bash
# usage: TREE=dev bash prof.sh GPU BATCH TAG [MODEL]   (detached; out $S/prof/TAG_bBATCH.*)
S=/data41/audreymao/perf7_20261105/exactplugin; PY=/data41/audreymao/a17_vllm/venv/bin/python
G=$1; B=$2; T=$3; M=${4:-$S/exactmodel}
TREE=${TREE:-repo} bash "$(dirname "$0")/r.sh" $G "mkdir -p $S/prof; nohup $PY experiments/perf/p7_exactplugin/prof_vllm.py --model $M --batch $B --out $S/prof/${T}_b$B --tag $T > $S/prof/${T}_b$B.log 2>&1 &"
