#!/bin/bash

python CDFKD.py \
--exp_name CDFKD-caltech101-resnet34-smhresnet18 \
--dataset caltech101 \
--model_t resnet34 \
--model_s smhresnet18 \
--epochs 300 \
--iters 50 \
--bsz 64 \
--lr_milestone 100 200 \
--weight_head 1 \
--weight_ens 1 \
--weight_bns 0.1 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1