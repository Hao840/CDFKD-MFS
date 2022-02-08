#!/bin/bash

python DFED.py \
--exp_name DFED-cifar100-resnet8x34-resnet8x18 \
--dataset cifar100 \
--model_t resnet8x34 \
--model_s resnet8x18 \
--epochs 300 \
--iters 50 \
--bsz 256 \
--lr_milestone 100 200 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1