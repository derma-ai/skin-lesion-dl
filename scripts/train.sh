#!/bin/bash

# The user provides the model to train with a BS of 32 and 64
python ./src/train.py -e 60 -b 32 -t "r,hflip,vflip" -osr 1.5 -l "wce" -ws 0 -ex "Without weighted sampling" -m "efficientnet_b0" -gpu $1
python ./src/train.py -e 60 -b 32 -t "r,hflip,vflip" -osr 1.5 -l "wce" -ws 3 -ex "Weighted sampling with 1.5 osr" -m "efficientnet_b0" -gpu $1
