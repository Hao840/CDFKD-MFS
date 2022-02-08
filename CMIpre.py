'''
Contrastive Model Inversion for Data-Free Knowledge Distillation - pre-invert
'''
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torch import optim
# from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm

from nets import gan_cmi
from registry import *
from utils import FeatureHook, InstanceMeanHook


class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)


class MemoryBank:
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn(max_size, dim_feat).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape

        self.data[self._ptr:self._ptr + n] = feat.detach()
        self._ptr = (self._ptr + n) % (self.max_size)
        self.n_updates += n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates > self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            # return self.data[:self._ptr]
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            try:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            except AttributeError:  # BN without affine
                pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')

    parser.add_argument('--g_steps', type=int, default=200, help='number of iterations for generation')
    parser.add_argument('--synthesis_bsz', type=int, default=256, help='batch size for synthesis')
    parser.add_argument('--lr_g', type=float, default=0.001)

    parser.add_argument('--bn', default=1, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--cr', default=0.8, type=float, help='scaling factor for contrastive model inversion')
    parser.add_argument('--cr_T', default=0.1, type=float, help='temperature for contrastive model inversion')

    parser.add_argument('--init_root', type=str, default='data/CMIpre', help='pre-inverted data path')
    parser.add_argument('--head_dim', type=int, default=256, help='hidden dim of the mlp head')
    parser.add_argument('--bank_size', type=int, default=40960, help='size of the memory bank')
    parser.add_argument('--n_neg', type=int, default=4096, help='number of sampled negative samples per iteration')

    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--id_t', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    num_classes = datainfo[args.dataset]['num_classes']
    img_size = datainfo[args.dataset]['img_size']
    target_num = datainfo[args.dataset]['train_num']
    mean = datainfo[args.dataset]['mean']
    std = datainfo[args.dataset]['std']
    if img_size == 32:
        Generator = gan_cmi.Generator
    else:
        Generator = gan_cmi.GeneratorLarge

    teacher = models[args.model_t](num_classes=num_classes).cuda()
    state_dict = torch.load(f'ckp/{args.dataset}-{args.model_t}-{args.id_t}.pt')['state']
    teacher.load_state_dict(state_dict)
    teacher.eval()

    # feature hook in the "dreaming to distill" paper
    hooks = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(FeatureHook(module))

    # hooks that collects features for contrastive learning
    cmi_hooks = []
    feature_layers = None  # use all conv layers
    if args.model_t in ['resnet8x34', 'resnet34']:  # only use blocks
        feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
    if feature_layers is not None:
        for layer in feature_layers:
            cmi_hooks.append(InstanceMeanHook(layer))
    else:
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                cmi_hooks.append(InstanceMeanHook(m))

    # collect the dimension of contrastive learning feature
    with torch.no_grad():
        teacher.eval()
        fake_inputs = torch.randn(1, 3, img_size, img_size).cuda()
        _ = teacher(fake_inputs)
        cmi_feature = torch.cat([h.instance_mean for h in cmi_hooks], dim=1)
        print(f'CMI dims: {cmi_feature.shape[1]}')
        del fake_inputs

    # build models, optimizers and schedulers
    generator = Generator(args.nz, img_size=img_size, nc=3).cuda()
    head = MLPHead(cmi_feature.shape[1], args.head_dim).cuda().train()

    optimizer_head = optim.Adam(head.parameters(), lr=args.lr_g)

    # bank for saving local and global features of contrastive learning
    mem_bank = MemoryBank('cpu', max_size=args.bank_size, dim_feat=2 * cmi_feature.shape[1])

    # data augmentation for generating positive and negative samples
    global_aug = transforms.Compose([
        transforms.RandomCrop(size=(img_size, img_size), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std)
    ])
    local_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.25, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std)
    ])

    data_dir = os.path.join(args.init_root, args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    current_num = len(os.listdir(data_dir))
    print(f'existing {current_num} imgs')

    while current_num < target_num:
        # Data synthesis
        z = torch.randn(args.synthesis_bsz, args.nz, requires_grad=True, device='cuda')
        targets = torch.randint(low=0, high=num_classes, size=(args.synthesis_bsz,)).cuda()

        reset_model(generator)
        optimizer_g = optim.Adam([{'params': generator.parameters()}, {'params': [z]}],
                                 args.lr_g, betas=(0.5, 0.999))

        best_cost = 1e6
        best_inputs = None
        for it in tqdm(range(args.g_steps), unit_scale=True,
                       desc=f'Data synthesizing [{current_num + 1}, {current_num + args.synthesis_bsz}]'):
            inputs = generator(z)
            global_view, local_view = global_aug(inputs), local_aug(inputs)

            # Inversion Loss
            t_out = teacher(global_view)
            loss_bn = args.bn * sum([h.r_feature for h in hooks])
            loss_oh = args.oh * F.cross_entropy(t_out, targets)
            loss_inv = loss_bn + loss_oh

            # Contrastive Loss
            global_feature = torch.cat([h.instance_mean for h in cmi_hooks], dim=1)
            _ = teacher(local_view)
            local_feature = torch.cat([h.instance_mean for h in cmi_hooks], dim=1)
            cached_feature, _ = mem_bank.get_data(args.n_neg)
            cached_local_feature, cached_global_feature = torch.chunk(cached_feature.cuda(), chunks=2, dim=1)

            proj_feature = head(
                torch.cat([local_feature, cached_local_feature, global_feature, cached_global_feature], dim=0))
            proj_local_feature, proj_global_feature = torch.chunk(proj_feature, chunks=2, dim=0)

            cr_logits = torch.mm(proj_local_feature,
                                 proj_global_feature.detach().T) / args.cr_T  # (N + N') x (N + N')
            cr_labels = torch.arange(start=0, end=len(cr_logits)).cuda()
            loss_cr = F.cross_entropy(cr_logits, cr_labels, reduction='none')  # (N + N')
            if mem_bank.n_updates > 0:
                loss_cr = loss_cr[:args.synthesis_bsz].mean() + loss_cr[args.synthesis_bsz:].mean()
            else:
                loss_cr = loss_cr.mean()

            loss = args.cr * loss_cr + loss_inv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    best_features = torch.cat([local_feature.data, global_feature.data], dim=1).data

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

        # save best inputs
        mem_bank.add(best_features)

        imgs = (best_inputs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(os.path.join(data_dir, f'{current_num + idx:05d}.png'))

        current_num += args.synthesis_bsz
