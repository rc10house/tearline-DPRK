#!/bin/bash 

set -e
ENVNAME=yolov8
export ENVDIR=$ENVNAME
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

python3 -m pip install torch torchvision 

