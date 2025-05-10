#!/bin/sh

python ../src/train.py \
    --batch_size 64 \
    --max_epochs 50 \
    --learning_rate 1e-4
