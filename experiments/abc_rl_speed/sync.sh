#!/usr/bin/env bash
# Copy code to 138: code_new = this working tree, code_old = src of exp/abc-rl (c756029) with this train.py on top.
H=audreymao@10.73.164.138; R="$(cd $(dirname $0)/../.. && pwd)"; SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=5"
$SSH $H 'mkdir -p ~/abc_rl_speed/code_new ~/abc_rl_speed/code_old'
rsync -a -e "$SSH" --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' $R/ $H:abc_rl_speed/code_new/
git -C $R archive exp/abc-rl | $SSH $H 'rm -rf ~/abc_rl_speed/code_old && mkdir -p ~/abc_rl_speed/code_old && tar -x -C ~/abc_rl_speed/code_old'
rsync -a -e "$SSH" $R/experiments/abc_rl/train.py $H:abc_rl_speed/code_old/experiments/abc_rl/train.py
