'''
Contrastive Model Inversion for Data-Free Knowledge Distillation
'''
import argparse
import functools
import math
import os
import random

import numpy as np
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
from utils import Recorder, FeatureHook, InstanceMeanHook, MultiMeter, kdloss, get_loader


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


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, init_root=None, transform=None, size=None):
        self.root = os.path.abspath(root)
        init_images = []
        if init_root is not None:
            init_images = self._collect_all_images(init_root)
        self.images = init_images + self._collect_all_images(self.root)
        if size is not None:
            self.images = self.images[-size:]
        self.transform = transform

    def _collect_all_images(self, root, postfix=('png', 'jpg', 'jpeg', 'JPEG')):
        images = []
        if isinstance(postfix, str):
            postfix = [postfix]
        for dirpath, dirnames, files in os.walk(root):
            for pos in postfix:
                for f in files:
                    if f.endswith(pos):
                        images.append(os.path.join(dirpath, f))
        return images

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)


class ImagePool(object):
    def __init__(self, root, init_root=None, size=None):
        self.root = os.path.abspath(root)
        self.init_root = os.path.abspath(init_root) if init_root is not None else None
        self.size = size
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs):
        save_image_batch(imgs, os.path.join(self.root, f'{self._idx}.png'), pack=False)
        self._idx += 1

    def get_dataset(self, transform=None):
        return UnlabeledImageDataset(self.root, self.init_root, transform, self.size)


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '-%d.png' % (idx))


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


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
    parser.add_argument('--exp_name', type=str, default='CMI-default')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')
    parser.add_argument('--model_s', type=str, default='resnet8x18')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--g_steps', type=int, default=200, help='number of iterations for generation')
    parser.add_argument('--kd_steps', type=int, default=2000, help='number of iterations for KD after generation')
    parser.add_argument('--ep_steps', type=int, default=2000, help='number of total iterations in each epoch')
    parser.add_argument('--sample_bsz', type=int, default=128, help='batch size for kd')
    parser.add_argument('--synthesis_bsz', type=int, default=256, help='batch size for synthesis')
    parser.add_argument('--lr_s', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--adv', default=0.5, type=float, help='scaling factor for adversarial distillation')
    parser.add_argument('--bn', default=1, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--cr', default=0.8, type=float, help='scaling factor for contrastive model inversion')
    parser.add_argument('--cr_T', default=0.1, type=float, help='temperature for contrastive model inversion')
    parser.add_argument('--T', default=1, type=float, help='temperature for kd')

    parser.add_argument('--init_root', type=str, default='data/CMIpre', help='pre-inverted data path')
    parser.add_argument('--head_dim', type=int, default=256, help='hidden dim of the mlp head')
    parser.add_argument('--bank_size', type=int, default=40960, help='size of the memory bank')
    parser.add_argument('--n_neg', type=int, default=4096, help='number of sampled negative samples per iteration')

    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--id_t', type=int, default=1)

    parser.add_argument('--ckp_freq', type=int, default=-1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    recorder = Recorder(base_path='result/main/CMI',
                        exp_name=args.exp_name,
                        logger_name=__name__,
                        code_path=__file__)

    recorder.logger.info(args)

    persistent_workers = True if args.num_workers > 0 else False
    kdloss_T = functools.partial(kdloss, T=args.T)

    num_classes = datainfo[args.dataset]['num_classes']
    img_size = datainfo[args.dataset]['img_size']
    train_num = datainfo[args.dataset]['train_num']
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
    student = models[args.model_s](num_classes=num_classes).cuda()
    generator = Generator(args.nz, img_size=img_size, nc=3).cuda()
    head = MLPHead(cmi_feature.shape[1], args.head_dim).cuda().train()

    optimizer_s = optim.SGD(student.parameters(), args.lr_s, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_head = optim.Adam(head.parameters(), lr=args.lr_g)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=args.epochs)

    # data pool for collecting generated images
    data_pool = ImagePool(root=os.path.join(recorder.exp_path, 'data'),
                          init_root=os.path.join(args.init_root, args.dataset),
                          size=train_num)

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

    # loaders of the original dataset for evaluation
    train_loader, test_loader = get_loader(args.dataset, 256, args.num_workers)
    test_num = len(test_loader.dataset)

    # meters for collecting logger info
    meter_train = MultiMeter()
    meter_train.register(['loss_s'])
    meter_test = MultiMeter()
    meter_test.register(['loss', 'acc'])

    for epoch in range(args.epochs):
        meter_train.reset()
        steps_in_epoch = args.ep_steps // args.kd_steps
        for step in range(steps_in_epoch):  # total kd_steps < ep_steps
            # 1. Data synthesis
            z = torch.randn(args.synthesis_bsz, args.nz, requires_grad=True, device='cuda')
            targets = torch.randint(low=0, high=num_classes, size=(args.synthesis_bsz,)).cuda()

            reset_model(generator)
            optimizer_g = optim.Adam([{'params': generator.parameters()}, {'params': [z]}],
                                     args.lr_g, betas=(0.5, 0.999))

            best_cost = 1e6
            best_inputs = None
            student.eval()
            for it in tqdm(range(args.g_steps), leave=False, unit_scale=True,
                           desc=f'Data synthesis in epoch-{epoch} step-{step}/{steps_in_epoch}'):
                inputs = generator(z)
                global_view, local_view = global_aug(inputs), local_aug(inputs)

                # Inversion Loss
                t_out = teacher(global_view)
                loss_bn = args.bn * sum([h.r_feature for h in hooks])
                loss_oh = args.oh * F.cross_entropy(t_out, targets)
                if args.adv > 0:
                    s_out = student(global_view)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1])
                    loss_adv = args.adv * -(kdloss(s_out, t_out, reduction='none').sum(1) * mask).mean()
                else:
                    loss_adv = loss_oh.new_zeros(1)
                loss_inv = loss_bn + loss_oh + loss_adv

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
                optimizer_head.zero_grad()
                loss.backward()
                optimizer_g.step()
                optimizer_head.step()

            # save best inputs and reset data iter
            data_pool.add(best_inputs)
            mem_bank.add(best_features)

            dst = data_pool.get_dataset(train_loader.dataset.transform)

            loader = torch.utils.data.DataLoader(
                dst, batch_size=args.sample_bsz, shuffle=True, num_workers=args.num_workers,
                persistent_workers=persistent_workers)
            data_iter = DataIter(loader)

            # 2. Knowledge distillation
            student.train()
            for i in tqdm(range(args.kd_steps), leave=False, unit_scale=True,
                          desc=f'KD in epoch-{epoch} step-{step}/{steps_in_epoch}'):
                images = data_iter.next().cuda()
                with torch.no_grad():
                    target = teacher(images)
                output = student(images)

                loss_s = kdloss_T(output, target)

                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

                meter_train.update('loss_s', loss_s.item())

        scheduler_s.step()

        recorder.logger.info(f'Epoch: {epoch}, '
                             f'loss_s: {meter_train.loss_s.avg:.6f}')

        recorder.add_scalars_from_dict({'loss_s': meter_train.loss_s.avg}, global_step=epoch)

        recorder.save_img(best_inputs[:25], f'{epoch}.png', nrow=5, normalize=True)

        meter_test.reset()
        student.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader, leave=False, unit_scale=True,
                                                      desc=f'Epoch-{epoch} test')):
                images = images.cuda()
                labels = labels.cuda()
                s_logits = student(images)

                meter_test.update('loss', F.cross_entropy(s_logits, labels).item(), n=len(images))
                pred = s_logits.data.max(1)[1]
                meter_test.update('acc', pred.eq(labels.data.view_as(pred)).sum().item())

        recorder.logger.info(f'Avg loss: {meter_test.loss.avg:.6f}, '
                             f'accuracy: {100 * meter_test.acc.sum / test_num:.2f}%')

        recorder.add_scalars_from_dict({'loss_test': meter_test.loss.avg,
                                        'accuracy': meter_test.acc.sum / test_num},
                                       global_step=epoch)

        if args.ckp_freq > 0 and epoch % args.ckp_freq == args.ckp_freq - 1:
            recorder.save_model({'student': student.state_dict(),
                                 'optimizer_s': optimizer_s.state_dict(),
                                 'scheduler_s': scheduler_s.state_dict()},
                                f'epoch{epoch}-ckp.pt')

    recorder.save_model(student.state_dict(), 'student.pt')
