#!/bin/bash

python DataTrain.py \
--exp_name DataTrain-caltech101-resnet34 \
--dataset caltech101 \
--model resnet34 \
--epochs 200 \
--bsz 64 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1