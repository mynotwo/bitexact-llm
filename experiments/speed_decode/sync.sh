#!/bin/bash
# rsync the worktree to 138:~/speed_decode/code (own directory; ~/speed belongs to another agent)
cd "$(dirname "$0")/../.." && ssh audreymao@10.73.164.138 'mkdir -p ~/speed_decode/code' && rsync -a --exclude .git ./ audreymao@10.73.164.138:speed_decode/code/ && scp -q experiments/speed_decode/env138.sh audreymao@10.73.164.138:speed_decode/env138.sh
