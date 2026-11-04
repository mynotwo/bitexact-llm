#!/bin/bash
# install.sh VER: fresh venv ~/portability/venv_VER with vllm==VER from the internal mirror (bytedpypi); existing envs untouched
V=$1; D=$HOME/portability/venv_$V; L=$HOME/portability/install_$V.log
export TMPDIR=$HOME/portability/tmp PIP_CACHE_DIR=$HOME/portability/pipcache
rm -rf $D; /usr/bin/python3 -m venv $D
$D/bin/pip install --timeout 60 vllm==$V numpy==2.3.5 --index-url https://bytedpypi.byted.org/simple/ > $L 2>&1 && echo INSTALLED_$V >> $L
