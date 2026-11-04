#!/usr/bin/env bash
# usage: sync.sh TAG [REV]  -- copy the working tree (or committed tree REV) to 138:~/speed2/code_TAG (scorer) and, with REV, to ~/speed2d/code_TAG (decode)
H=audreymao@10.73.164.138; R="$(cd $(dirname $0)/../.. && pwd)"
ssh -n $H "rm -rf ~/speed2/code_$1 ~/speed2d/code_$1; mkdir -p ~/speed2/code_$1 ~/speed2d/code_$1"
if [ -n "$2" ]; then
  git -C $R archive $2 | ssh $H "tar -x -C ~/speed2/code_$1; tar -c -C ~/speed2/code_$1 . | tar -x -C ~/speed2d/code_$1"
else
  for d in speed2/code_$1 speed2d/code_$1; do rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' $R/ $H:$d/; done
fi
scp -q $R/experiments/speed/env_decode.sh $H:speed2d/env_decode.sh
