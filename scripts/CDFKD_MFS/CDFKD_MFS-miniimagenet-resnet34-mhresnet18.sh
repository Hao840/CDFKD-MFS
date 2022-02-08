#!/bin/bash

python CDFKD_MFS.py \
--exp_name CDFKD_MFS-miniimagenet-resnet34-mhresnet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s mhresnet18 \
--epochs 100 \
--iters 750 \
--bsz 64 \
--lr_milestone 30 60 90 \
--weight_feat 0.05 \
--weight_ens 1 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1