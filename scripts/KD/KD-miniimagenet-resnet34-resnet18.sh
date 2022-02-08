#!/bin/bash

python KD.py \
--exp_name KD-miniimagenet-resnet34-resnet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s resnet18 \
--epochs 100 \
--bsz 64 \
--lr 0.1 \
--lr_milestone 30 60 90 \
--num_workers 4 \
--ckp_freq -1