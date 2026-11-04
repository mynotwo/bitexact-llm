#!/usr/bin/env bash
# Host 138: rebuild the int8 constants from the original checkpoint (deterministic; sha must equal the 141 one) and the dequantized checkpoint.
set -e
W=$HOME/abc_measure; cd $W
export HF_HUB_OFFLINE=1 TMPDIR=$W/tmp HF_HOME=$W/hf PYTHONPATH=$W/code/src
PY=$W/venv/bin/python
ORIG=$(ls -d $W/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/*)
$PY code/examples/build_constants.py --model $ORIG --mode int8 --out const.pt
$PY code/experiments/abc_measure/export_dequant.py --const const.pt --orig $ORIG --out deqmodel
touch $W/PREPDONE
echo PREPDONE
