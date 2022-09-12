#!/bin/bash

# The user provides the model to train with a BS of 32 and 64
python ./../src/train.py -e 500 -b 32 -d "preprocessed" -t "r,hflip,vflip" -osr 1.5 -l "wce" -ws 1 -ex "Classic weighted sampling on preprocessed images" -m "efficientnet_b0"
python ./../src/train.py -e 200 -b 32 -d "preprocessed" -t "r,hflip,vflip,norm" -osr 1.5 -l "wce" -ws 1 -ex "Classic weighted sampling with Normalization = [mean=[0.657, 0.548, 0.532], std=[0.204, 0.197, 0.208]] over preprocessed dataset" -m "efficientnet_b0"
python ./../src/train.py -e 200 -b 32 -d "original" -t "r,hflip,vflip,norm" -osr 1.5 -l "wce" -ws 1 -ex "Classic weighted sampling with Normalization = [mean=[0.624, 0.520, 0.504], std=[0.242, 0.223, 0.231]] over original dataset." -m "efficientnet_b0"
