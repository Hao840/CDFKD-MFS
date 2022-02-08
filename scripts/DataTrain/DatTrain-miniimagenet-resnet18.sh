#!/bin/bash

python DataTrain.py \
--exp_name DataTrain-miniimagenet-resnet18 \
--dataset miniimagenet \
--model resnet18 \
--epochs 100 \
--bsz 64 \
--lr 0.1 \
--lr_milestone 50 80 \
--num_workers 4 \
--ckp_freq -1