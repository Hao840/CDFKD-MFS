#!/bin/bash

python KD.py \
--exp_name KD-cifar100-wrn4x402-resnet8x18 \
--dataset cifar100 \
--model_t wrn4x402 \
--model_s resnet8x18 \
--epochs 200 \
--bsz 128 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1