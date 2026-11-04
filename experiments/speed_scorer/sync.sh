#!/usr/bin/env bash
# usage: sync.sh TAG [REV]  -- copy the working tree (or the committed tree REV) to 138:~/speed/code_TAG
H=audreymao@10.73.164.138
ssh -n $H "rm -rf ~/speed/code_$1; mkdir -p ~/speed/code_$1"
if [ -n "$2" ]; then git -C "$(dirname $0)/../.." archive "$2" | ssh $H "tar -x -C ~/speed/code_$1"
else rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' "$(dirname $0)/../../" $H:speed/code_$1/; fi
