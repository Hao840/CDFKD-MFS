#!/bin/bash

python CDFKD.py \
--exp_name CDFKD-miniimagenet-resnet34-smhresnet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s smhresnet18 \
--epochs 100 \
--iters 750 \
--bsz 64 \
--lr_milestone 30 60 90 \
--weight_head 1 \
--weight_ens 1 \
--weight_bns 0.1 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1