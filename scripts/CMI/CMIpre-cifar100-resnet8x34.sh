#!/bin/bash

python CMIpre.py \
--dataset cifar100 \
--model_t resnet8x34 \
--g_steps 200 \
--synthesis_bsz 256 \
--lr_g 0.001