#!/bin/bash

python CMIpre.py \
--dataset caltech101 \
--model_t resnet34 \
--g_steps 200 \
--synthesis_bsz 256 \
--lr_g 0.001