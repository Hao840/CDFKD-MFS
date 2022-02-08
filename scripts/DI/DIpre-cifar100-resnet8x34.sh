#!/bin/bash

python DIpre.py \
--dataset cifar100 \
--model_t resnet8x34 \
--iters 2000 \
--bsz 256 \
--lr 0.1 \
--betas 0.9 0.999 \
--jitter 2 \
--first_bn_multiplier 1 \
--main_loss_multiplier 1 \
--var_scale_l1 0 \
--var_scale_l2 0.001 \
--bn_reg_scale 10 \
--l2_scale 0

