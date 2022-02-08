#!/bin/bash

python KD.py \
--exp_name KD-cifar100-wrn4x402-wrn4x162 \
--dataset cifar100 \
--model_t wrn4x402 \
--model_s wrn4x162 \
--epochs 200 \
--bsz 128 \
--lr 0.1 \
--lr_milestone 80 120 \
--num_workers 4 \
--ckp_freq -1