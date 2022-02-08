#!/bin/bash

python DIpre.py \
--dataset miniimagenet \
--model_t resnet34 \
--iters 2000 \
--bsz 128 \
--lr 0.25 \
--betas 0.5 0.9 \
--jitter 30 \
--first_bn_multiplier 10 \
--main_loss_multiplier 1 \
--var_scale_l1 0 \
--var_scale_l2 0.0001 \
--bn_reg_scale 0.01 \
--l2_scale 0.00001 \
--lr_schedule \
--multi_rs \
--do_clip \
--do_flip
