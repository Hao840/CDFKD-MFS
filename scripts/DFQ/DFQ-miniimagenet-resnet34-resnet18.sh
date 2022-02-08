#!/bin/bash

python DFQ.py \
--exp_name DFQ-miniimagenet-resnet34-resnet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s resnet18 \
--epochs 200 \
--iters 750 \
--bsz 64 \
--num_workers 4 \
--ckp_freq -1