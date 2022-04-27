#!/bin/sh

python3 ../src/train.py \
    --batch_size 64 \
    --max_epochs 2 \
    --learning_rate 1e-4
