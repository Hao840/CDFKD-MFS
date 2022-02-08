#!/bin/bash

python CDFKD.py \
--exp_name CDFKD-cifar100-resnet8x34-smhresnet8x18 \
--dataset cifar100 \
--model_t resnet8x34 \
--model_s smhresnet8x18 \
--epochs 300 \
--iters 50 \
--bsz 256 \
--lr_milestone 100 200 \
--weight_head 1 \
--weight_ens 1 \
--weight_bns 0.1 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1