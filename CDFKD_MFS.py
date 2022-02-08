import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from registry import *
from utils import Recorder, FeatureHook, multi_forward, MultiMeter, get_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='CDFKDNew-default')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')
    parser.add_argument('--model_s', type=str, default='mhresnet8x18')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--lr_s', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.001)
    parser.add_argument('--lr_milestone', type=int, nargs='+', default=[100, 200])

    parser.add_argument('--weight_s', type=float, default=1)
    parser.add_argument('--weight_feat', type=float, default=0.2)
    parser.add_argument('--weight_ens', type=float, default=5)

    parser.add_argument('--weight_g', type=float, default=1)
    parser.add_argument('--weight_bns', type=float, default=0.1)

    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--num_t', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--ckp_freq', type=int, default=-1)
    parser.add_argument('--no_img_saving', action='store_true')

    return parser.parse_args()


def criterion_feat(t_f, s_f):
    delta = torch.abs(s_f - t_f)
    loss_feat = torch.mean(delta.sum(1))
    return loss_feat


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    recorder = Recorder(base_path='result/main/CDFKDNew',
                        exp_name=args.exp_name,
                        logger_name=__name__,
                        code_path=__file__)

    recorder.logger.info(args)

    num_classes = datainfo[args.dataset]['num_classes']
    img_size = datainfo[args.dataset]['img_size']
    Generator = datainfo[args.dataset]['generator']

    teachers = []
    for i in range(1, args.num_t + 1):
        teacher = models[args.model_t](num_classes=num_classes).cuda()
        state_dict = torch.load(f'ckp/{args.dataset}-{args.model_t}-{i}.pt')['state']
        teacher.load_state_dict(state_dict)
        teacher.eval()
        teachers.append(teacher)
    del state_dict

    hooks = []
    for teacher in teachers:
        feature_hook = []
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                feature_hook.append(FeatureHook(module))
        hooks.append(feature_hook)

    student = models[args.model_s](headers=args.num_t, num_classes=num_classes).cuda()

    generator = Generator(args.nz, img_size=img_size, nc=3).cuda()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_s = optim.SGD(student.parameters(), lr=args.lr_s, momentum=0.9, weight_decay=5e-4)

    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, args.lr_milestone, 0.1)
    scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, args.lr_milestone, 0.1)

    _, test_loader = get_loader(args.dataset, 256, args.num_workers)
    test_num = len(test_loader.dataset)

    meter_train = MultiMeter()
    meter_train.register(['loss_s', 'loss_feat', 'loss_ens', 'loss_g', 'loss_bns'])
    meter_test = MultiMeter()
    meter_test.register(['loss', 'acc_ens'] + [f'acc_head_{i}' for i in range(args.num_t)])

    rescale = [5] + [1. for _ in range(len(hooks[0]) - 1)]
    for epoch in range(args.epochs):
        meter_train.reset()
        for _ in tqdm(range(args.iters), leave=False, unit_scale=True, desc=f'Epoch-{epoch} train'):
            student.train()
            generator.train()
            # train the student
            for i in range(5):
                z = torch.randn(args.bsz, args.nz).cuda()
                with torch.no_grad():
                    gen_imgs = generator(z)
                    t_logits, t_feats = multi_forward(teachers, gen_imgs, out_feature=True)
                s_logits, s_feats = student(gen_imgs, out_feature=True)

                # adversarial loss
                loss_s = args.weight_s * torch.stack(
                    [F.l1_loss(s_logit, t_logit.detach()) for s_logit, t_logit in zip(s_logits, t_logits)]).mean()

                # feature loss
                if args.weight_feat == 0:
                    loss_feat = torch.tensor(0).float().cuda()
                else:
                    loss_feat = args.weight_feat * torch.stack(
                        [criterion_feat(s_feat, t_feat.detach()) for s_feat, t_feat in zip(s_feats, t_feats)]).mean()

                # ensemble loss
                loss_ens = args.weight_ens * F.l1_loss(torch.stack(s_logits).mean(0), torch.stack(t_logits).mean(0))

                optimizer_s.zero_grad()
                # weighted loss
                (loss_s + loss_feat + loss_ens).backward()
                optimizer_s.step()

                meter_train.update('loss_s', loss_s.item())
                meter_train.update('loss_feat', loss_feat.item())
                meter_train.update('loss_ens', loss_ens.item())

            # train the generator
            z = torch.randn(args.bsz, args.nz).cuda()
            optimizer_g.zero_grad()
            gen_imgs = generator(z)
            s_logits = student(gen_imgs)

            # forward pass through a single teacher each time to save GPU memory
            for i in range(args.num_t):
                t_logit = teachers[i](gen_imgs)

                # adverserial loss
                loss_g = args.weight_g * -(F.l1_loss(s_logits[i], t_logit) / args.num_t)

                # BN loss
                loss_bns = args.weight_bns * \
                           (sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hooks[i])]) / args.num_t)

                # weighted loss
                (loss_g + loss_bns).backward(retain_graph=True)

                meter_train.update('loss_g', loss_g.item())
                meter_train.update('loss_bns', loss_bns.item())

                t_logit = None
                loss_bns = None
                loss_g = None
                loss = None
                for hook in hooks[i]:
                    hook.r_feature = None

            optimizer_g.step()

        scheduler_g.step()
        scheduler_s.step()

        recorder.logger.info(f'Epoch: {epoch}, '
                             f'loss_s: {meter_train.loss_s.avg:.6f}, '
                             f'loss_ens: {meter_train.loss_ens.avg:.6f}, '
                             f'loss_feat: {meter_train.loss_feat.avg:.6f} | '
                             f'loss_g: {meter_train.loss_g.avg:.6f}, '
                             f'loss_bns: {meter_train.loss_bns.avg:.6f}')

        recorder.add_scalars_from_dict({'loss_s': meter_train.loss_s.avg,
                                        'loss_feat': meter_train.loss_ens.avg,
                                        'loss_ens': meter_train.loss_feat.avg,
                                        'loss_g': meter_train.loss_g.avg,
                                        'loss_bns': meter_train.loss_bns.avg},
                                       global_step=epoch)

        if not args.no_img_saving:
            recorder.save_img(gen_imgs[:25], f'{epoch}.png', nrow=5, normalize=True)

        meter_test.reset()
        student.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader, leave=False, unit_scale=True,
                                                      desc=f'Epoch-{epoch} test')):
                images = images.cuda()
                labels = labels.cuda()
                s_logits = student(images)
                s_logits_mean = torch.stack(s_logits).mean(0)

                meter_test.update('loss', F.cross_entropy(s_logits_mean, labels).item(), n=len(images))
                pred = s_logits_mean.data.max(1)[1]
                meter_test.update('acc_ens', pred.eq(labels.data.view_as(pred)).sum().item())

                for h, s_logit_i in enumerate(s_logits):
                    pred = s_logit_i.data.max(1)[1]
                    meter_test.update(f'acc_head_{h}', pred.eq(labels.data.view_as(pred)).sum().item())

        branch_acc = [meter_test[f'acc_head_{i}'].sum / test_num for i in range(args.num_t)]
        branch_info = ' / '.join(f'{100 * branch_acc_i:.2f}%' for branch_acc_i in branch_acc)
        recorder.logger.info(f'Avg loss: {meter_test.loss.avg:.6f}, '
                             f'accuracy: {100 * meter_test.acc_ens.sum / test_num:.2f}% | '
                             f'({branch_info})')

        recorder.add_scalars_from_dict({'loss_test': meter_test.loss.avg,
                                        'accuracy': meter_test.acc_ens.sum / test_num},
                                       global_step=epoch)

        if args.ckp_freq > 0 and epoch % args.ckp_freq == args.ckp_freq - 1:
            recorder.save_model({'generator': generator.state_dict(),
                                 'student': student.state_dict(),
                                 'optimizer_s': optimizer_s.state_dict(),
                                 'optimizer_g': optimizer_g.state_dict(),
                                 'scheduler_s': scheduler_s.state_dict(),
                                 'scheduler_g': scheduler_g.state_dict()},
                                f'epoch{epoch}-ckp.pt')

    recorder.save_model(generator.state_dict(), 'generator.pt')
    recorder.save_model(student.state_dict(), 'student.pt')
