#!/bin/bash

python CDFKD_MFS.py \
--exp_name CDFKD_MFS-caltech101-resnet34-mhresnet18 \
--dataset caltech101 \
--model_t resnet34 \
--model_s mhresnet18 \
--epochs 300 \
--iters 50 \
--bsz 64 \
--lr_milestone 100 200 \
--weight_feat 0.2 \
--weight_ens 5 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1