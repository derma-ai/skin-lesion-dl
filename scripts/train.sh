#!/bin/bash

# The user provides the model to train with a BS of 32 and 64
python ./src/train.py -e 200 -b 32 -t "r,hflip,vflip" -osr 1.5 -l "wce" -ws 1 -ex "Classic weighted sampling with color constancy base transform" -m "efficientnet_b0"
