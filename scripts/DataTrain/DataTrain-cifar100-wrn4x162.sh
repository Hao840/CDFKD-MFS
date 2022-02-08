#!/bin/bash

python DataTrain.py \
--exp_name DataTrain-cifar100-wrn4x162 \
--dataset cifar100 \
--model wrn4x162 \
--epochs 200 \
--bsz 128 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1