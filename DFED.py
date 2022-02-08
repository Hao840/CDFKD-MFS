'''
Data-F ree Ensemble Knowledge Distillation for Privacy-conscious Multimedia Model Compression
'''
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
    parser.add_argument('--exp_name', type=str, default='DFED-default')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')
    parser.add_argument('--model_s', type=str, default='resnet8x18')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--bsz', type=int, default=256)
    parser.add_argument('--lr_s', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.001)
    parser.add_argument('--lr_milestone', type=int, nargs='+', default=[100, 200])

    parser.add_argument('--weight_s', type=float, default=1)
    parser.add_argument('--weight_g', type=float, default=1)
    parser.add_argument('--weight_bns', type=float, default=0.1)

    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--num_t', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--ckp_freq', type=int, default=-1)
    parser.add_argument('--no_img_saving', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    recorder = Recorder(base_path='result/main/DFED',
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

    student = models[args.model_s](num_classes=num_classes).cuda()

    generator = Generator(args.nz, img_size=img_size, nc=3).cuda()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_s = optim.SGD(student.parameters(), lr=args.lr_s, momentum=0.9, weight_decay=5e-4)

    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, args.lr_milestone, 0.1)
    scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, args.lr_milestone, 0.1)

    _, test_loader = get_loader(args.dataset, 256, args.num_workers)
    test_num = len(test_loader.dataset)

    meter_train = MultiMeter()
    meter_train.register(['loss_s', 'loss_g', 'loss_bns'])
    meter_test = MultiMeter()
    meter_test.register(['loss', 'acc'])

    rescale = [5] + [1. for _ in range(len(hooks[0]) - 1)]
    for epoch in range(args.epochs):
        meter_train.reset()
        student.train()
        generator.train()
        for _ in tqdm(range(args.iters), leave=False, unit_scale=True, desc=f'Epoch-{epoch} train'):
            # train the student
            for i in range(5):
                z = torch.randn(args.bsz, args.nz).cuda()
                with torch.no_grad():
                    gen_imgs = generator(z)
                    t_logits = torch.stack(multi_forward(teachers, gen_imgs)).mean(0)
                s_logits = student(gen_imgs)

                # adversarial loss
                loss_s = args.weight_s * F.l1_loss(s_logits, t_logits.detach())

                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

                meter_train.update('loss_s', loss_s.item())

            # train the generator
            z = torch.randn(args.bsz, args.nz).cuda()
            gen_imgs = generator(z)
            t_logits = torch.stack(multi_forward(teachers, gen_imgs)).mean(0)
            s_logits = student(gen_imgs)

            # adverserial loss
            loss_g = args.weight_g * -F.l1_loss(s_logits, t_logits)

            # BN loss
            loss_bns = sum(sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(hooks[i])]) for i in range(3))
            loss_bns = args.weight_bns * loss_bns / args.num_t

            optimizer_g.zero_grad()
            # weighted loss
            (loss_g + loss_bns).backward()
            optimizer_g.step()

            meter_train.update('loss_g', loss_g.item())
            meter_train.update('loss_bns', loss_bns.item())

        scheduler_g.step()
        scheduler_s.step()

        recorder.logger.info(f'Epoch: {epoch}, '
                             f'loss_s: {meter_train.loss_s.avg:.6f} | '
                             f'loss_g: {meter_train.loss_g.avg:.6f}, '
                             f'loss_bns: {meter_train.loss_bns.avg:.6f}')

        recorder.add_scalars_from_dict({'loss_s': meter_train.loss_s.avg,
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

                meter_test.update('loss', F.cross_entropy(s_logits, labels).item(), n=len(images))
                pred = s_logits.data.max(1)[1]
                meter_test.update('acc', pred.eq(labels.data.view_as(pred)).sum().item())

        recorder.logger.info(f'Avg loss: {meter_test.loss.avg:.6f}, '
                             f'accuracy: {100 * meter_test.acc.sum / test_num:.2f}%')

        recorder.add_scalars_from_dict({'loss_test': meter_test.loss.avg,
                                        'accuracy': meter_test.acc.sum / test_num},
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
