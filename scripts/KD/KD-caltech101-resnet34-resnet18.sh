#!/bin/bash

python KD.py \
--exp_name KD-caltech101-resnet34-resnet18 \
--dataset caltech101 \
--model_t resnet34 \
--model_s resnet18 \
--epochs 200 \
--bsz 64 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1