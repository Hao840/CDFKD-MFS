#!/bin/bash

python CDFKD_MFS.py \
--exp_name CDFKD_MFS-cifar100-wrn4x402-mhresnet8x18 \
--dataset cifar100 \
--model_t wrn4x402 \
--model_s mhresnet8x18 \
--epochs 300 \
--iters 50 \
--bsz 256 \
--lr_milestone 100 200 \
--weight_feat 0 \
--weight_ens 5 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1