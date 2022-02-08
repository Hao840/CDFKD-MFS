#!/bin/bash

python DFAD.py \
--exp_name DFAD-cifar100-resnet8x18-wrn4x402 \
--dataset cifar100 \
--model_t resnet8x18 \
--model_s wrn4x402 \
--epochs 300 \
--iters 50 \
--bsz 256 \
--lr_milestone 100 200 \
--num_workers 4 \
--ckp_freq -1