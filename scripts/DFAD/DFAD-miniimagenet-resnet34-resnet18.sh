#!/bin/bash

python DFAD.py \
--exp_name DFAD-miniimagenet-resnet34-resnet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s resnet18 \
--epochs 100 \
--iters 750 \
--bsz 64 \
--lr_milestone 30 60 90 \
--num_workers 4 \
--ckp_freq -1