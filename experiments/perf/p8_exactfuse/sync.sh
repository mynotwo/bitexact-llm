#!/usr/bin/env bash
# Copy HEAD (plus uncommitted tracked/untracked changes) of the worktree to 141. usage: bash sync.sh
set -e
HOST=audreymao@10.154.162.141
S=/data41/audreymao/perf8_20261105/exactfuse
LOCAL=$(cd "$(dirname "$0")/../../.." && pwd)
ssh $HOST "mkdir -p $S/${TREE:-repo}"
rsync -a --delete --exclude .git --exclude runs "$LOCAL/" $HOST:$S/${TREE:-repo}/
echo synced
