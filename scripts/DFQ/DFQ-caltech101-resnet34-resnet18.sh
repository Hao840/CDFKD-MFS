#!/bin/bash

python DFQ.py \
--exp_name DFQ-caltech101-resnet34-resnet18 \
--dataset caltech101 \
--model_t resnet34 \
--model_s resnet18 \
--epochs 200 \
--iters 400 \
--bsz 64 \
--num_workers 4 \
--ckp_freq -1