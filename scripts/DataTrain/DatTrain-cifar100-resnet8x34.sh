#!/bin/bash

python DataTrain.py \
--exp_name DataTrain-cifar100-resnet8x34 \
--dataset cifar100 \
--model resnet8x34 \
--epochs 200 \
--bsz 128 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1