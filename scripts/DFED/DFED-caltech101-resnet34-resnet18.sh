#!/bin/bash

python DFED.py \
--exp_name DFED-caltech101-resnet34-resnet18 \
--dataset caltech101 \
--model_t resnet34 \
--model_s resnet18 \
--epochs 300 \
--iters 50 \
--bsz 64 \
--lr_milestone 100 200 \
--num_t 3 \
--num_workers 4 \
--ckp_freq -1