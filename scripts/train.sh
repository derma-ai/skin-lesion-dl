#!/bin/bash
# The user provides the model to train with a BS of 32 and 64
python3 ../src/train.py -e 30 -b 32 -t "r,hflip,vflip" -osr 1.5 -l "wce" -ex "Weighted sampling with 1.5 osr, weighted loss, no normalization, transforms" -m $1 -gpu $2
python3 ../src/train.py -e 30 -b 64 -t "r,hflip,vflip" -osr 1.5 -l "wce" -ex "Weighted sampling with 1.5 osr, weighted loss, no normalization, transforms" -m $1 -gpu $2