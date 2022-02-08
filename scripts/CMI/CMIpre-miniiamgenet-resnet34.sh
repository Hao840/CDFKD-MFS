#!/bin/bash

python CMI.py \
--exp_name CMI-miniimagenet-resnet34-resmet18 \
--dataset miniimagenet \
--model_t resnet34 \
--model_s resnet18 \
--epochs 40 \
--g_steps 200 \
--kd_steps 2000  \
--ep_steps 2000  \
--sample_bsz 128 \
--synthesis_bsz 64 \
--adv 0.5 \
--bn 1 \
--oh 0.5 \
--cr 0.8 \
--cr_T 0.1 \
--T 20