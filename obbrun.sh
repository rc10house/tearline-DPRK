#!/bin/bash 

set -e
ENVNAME=yolov8
export ENVDIR=$ENVNAME
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

tar -xzf dota_dataset.tar.gz
mkdir datasets && mv dataset datasets

python3 -m pip install torch torchvision 


python3 obbtrain.py 

tar -czf export_obb.tar.gz yolov8_obb2
