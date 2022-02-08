'''
Dreaming to Distill Data-free Knowledge Transfer via DeepInversion
step 1. generate samples
'''
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torch import optim
from tqdm import tqdm

from registry import *
from utils import FeatureHook


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def clip(image_tensor, mean, std):
    '''
    adjust the input based on mean and variance
    '''
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/DI')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')
    parser.add_argument('--id_t', type=int, default=1)

    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--betas', type=float, nargs='+', default=(0.9, 0.999))

    parser.add_argument('--jitter', type=int, default=2)
    parser.add_argument('--first_bn_multiplier', default=1, type=float)
    parser.add_argument('--main_loss_multiplier', default=1, type=float)
    parser.add_argument('--var_scale_l1', default=0, type=float)
    parser.add_argument('--var_scale_l2', default=0.001, type=float)
    parser.add_argument('--bn_reg_scale', default=10, type=float)
    parser.add_argument('--l2_scale', default=0, type=float)

    parser.add_argument('--lr_schedule', action='store_true')
    parser.add_argument('--multi_rs', action='store_true')
    parser.add_argument('--do_clip', action='store_true')
    parser.add_argument('--do_flip', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    num_classes = datainfo[args.dataset]['num_classes']
    img_size = datainfo[args.dataset]['img_size']
    mean = datainfo[args.dataset]['mean']
    std = datainfo[args.dataset]['std']
    target_num = datainfo[args.dataset]['train_num']

    teacher = models[args.model_t](num_classes=num_classes).cuda()
    state_dict = torch.load(f'ckp/{args.dataset}-{args.model_t}-{args.id_t}.pt')['state']
    teacher.load_state_dict(state_dict)
    teacher.eval()

    hooks = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(FeatureHook(module))

    pooling_function = nn.AvgPool2d(kernel_size=2)

    data_dir = os.path.join(args.root, args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    current_num = len(os.listdir(data_dir))
    print(f'existing {current_num} imgs')

    while current_num < target_num:
        inputs = torch.randn((args.bsz, 3, img_size, img_size), requires_grad=True, device='cuda')
        targets = torch.randint(low=0, high=num_classes, size=(args.bsz,)).cuda()

        best_cost = 1e4
        best_inputs = None
        with tqdm(total=args.iters, unit_scale=True,
                  desc=f'Data synthesizing [{current_num + 1}, {current_num + args.bsz}]') as pbar:
            for lr_it, lower_res in enumerate([2, 1]):
                if lr_it == 0:
                    iterations_per_layer = args.iters // 2
                else:
                    iterations_per_layer = args.iters // 2 if args.multi_rs else args.iters

                if lr_it == 0 and not args.multi_rs:
                    continue

                lim_0, lim_1 = args.jitter // lower_res, args.jitter // lower_res

                optimizer = optim.Adam([inputs], lr=args.lr, betas=args.betas, eps=1e-8)

                lr_scheduler = lr_cosine_policy(args.lr, 100, iterations_per_layer)

                for iteration_loc in range(iterations_per_layer):
                    # learning rate scheduling
                    if args.lr_schedule:
                        lr_scheduler(optimizer, iteration_loc, iteration_loc)

                    # perform downsampling if needed
                    if lower_res != 1:
                        inputs_jit = pooling_function(inputs)
                    else:
                        inputs_jit = inputs

                    # apply random jitter offsets
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                    # Flipping
                    flip = random.random() > 0.5
                    if flip and args.do_flip:
                        inputs_jit = torch.flip(inputs_jit, dims=(3,))

                    # forward pass
                    outputs = teacher(inputs_jit)

                    # R_cross classification loss
                    loss = F.cross_entropy(outputs, targets)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # R_feature loss
                    rescale = [args.first_bn_multiplier] + [1. for _ in range(len(hooks) - 1)]
                    loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hooks)])

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(args.bsz, -1), dim=1).mean()

                    # combining losses
                    loss_aux = args.var_scale_l1 * loss_var_l1 + \
                               args.var_scale_l2 * loss_var_l2 + \
                               args.bn_reg_scale * loss_r_feature + \
                               args.l2_scale * loss_l2

                    loss = args.main_loss_multiplier * loss + loss_aux

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # clip color outlayers
                    if args.do_clip:
                        inputs.data = clip(inputs.data, mean, std)

                    if best_cost > loss.item() or best_inputs is None:
                        best_inputs = inputs.data.clone()
                        best_cost = loss.item()

                    pbar.update(1)

        low = float(best_inputs.min())
        high = float(best_inputs.max())
        best_inputs.clamp_(min=low, max=high)
        best_inputs.sub_(low).div_(max(high - low, 1e-5))
        for idx, img in enumerate(best_inputs):
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(data_dir, f'{current_num + idx:05d}.png'))

        current_num += args.bsz
