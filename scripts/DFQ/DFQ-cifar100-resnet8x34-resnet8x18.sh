#!/bin/bash

python DFQ.py \
--exp_name DFQ-cifar100-resnet8x34-resnet8x18 \
--dataset cifar100 \
--model_t resnet8x34 \
--model_s resnet8x18 \
--epochs 200 \
--iters 400 \
--bsz 256 \
--num_workers 4 \
--ckp_freq -1