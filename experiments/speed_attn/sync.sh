#!/bin/bash
# rsync the worktree to 138:~/speed_attn/code (own directory; ~/speed belongs to another agent)
cd "$(dirname "$0")/../.." && ssh audreymao@10.73.164.138 'mkdir -p ~/speed_attn/code' && rsync -a --exclude .git ./ audreymao@10.73.164.138:speed_attn/code/ && scp -q experiments/speed_attn/env138.sh audreymao@10.73.164.138:speed_attn/env138.sh
