#!/usr/bin/env bash
# usage: sync.sh TAG  -- copy the working tree to 138:~/speed/code_TAG (one snapshot per measured item)
H=audreymao@10.73.164.138
ssh -n $H "mkdir -p ~/speed/code_$1"
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' "$(dirname $0)/../../" $H:speed/code_$1/
