#!/usr/bin/env bash

echo "Run test.sh"
python main_func.py \
    --cuda 3 \
    --save_model True \
    -t synthetic



echo "End test.sh"

