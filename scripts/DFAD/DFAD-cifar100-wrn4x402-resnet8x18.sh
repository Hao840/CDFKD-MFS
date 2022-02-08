#!/bin/bash

python DFAD.py \
--exp_name DFAD-cifar100-wrn4x402-resnet8x18 \
--dataset cifar100 \
--model_t wrn4x402 \
--model_s resnet8x18 \
--epochs 300 \
--iters 50 \
--bsz 256 \
--lr_milestone 100 200 \
--num_workers 4 \
--ckp_freq -1