#!/usr/bin/env bash
# usage: sync.sh TAG  -- copy the working tree to 138:~/speed2/code_TAG (scorer) and ~/speed2d/code (decode)
H=audreymao@10.73.164.138; R="$(dirname $0)/../.."
ssh -n $H "rm -rf ~/speed2/code_$1; mkdir -p ~/speed2/code_$1 ~/speed2d/code"
rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' $R/ $H:speed2/code_$1/
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' $R/ $H:speed2d/code/
scp -q $R/experiments/speed/env_decode.sh $H:speed2d/env_decode.sh
