import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from registry import *
from utils import Recorder, MultiMeter, get_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='DataTrain-default')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='resnet8x34')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bsz', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_milestone', type=int, nargs='+', default=[80, 120])
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--ckp_freq', type=int, default=-1)

    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    recorder = Recorder(base_path='result/main/DataTrain',
                        exp_name=args.exp_name,
                        logger_name=__name__,
                        code_path=__file__)
    recorder.logger.info(args)

    num_classes = datainfo[args.dataset]['num_classes']
    net = models[args.model](num_classes=num_classes).cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestone, 0.1)

    train_loader, test_loader = get_loader(args.dataset, args.bsz, args.num_workers)
    train_num = len(train_loader.dataset)
    test_num = len(test_loader.dataset)

    meter_train = MultiMeter()
    meter_train.register(['loss'])
    meter_test = MultiMeter()
    meter_test.register(['loss', 'acc'])

    for epoch in range(args.epochs):
        meter_train.reset()
        net.train()
        for images, labels in tqdm(train_loader, leave=False, unit_scale=True, desc=f'Epoch-{epoch} train'):
            images, labels = images.cuda(), labels.cuda()
            output = net(images)
            loss = F.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter_train.update('loss', loss.item(), len(images))

        recorder.logger.info(f'Train - Epoch {epoch}, Loss: {meter_train.loss.avg:.6f}')
        recorder.writer.add_scalar('loss_train', meter_train.loss.avg, epoch)

        scheduler.step()

        if epoch % args.test_freq == args.test_freq - 1 or epoch >= args.epochs - 10:
            meter_test.reset()
            net.eval()
            with torch.no_grad():
                for images, labels in tqdm(test_loader, leave=False, unit_scale=True, desc=f'Epoch-{epoch} test'):
                    images, labels = images.cuda(), labels.cuda()
                    output = net(images)

                    meter_test.update('loss', F.cross_entropy(output, labels).item(), n=len(images))
                    pred = output.data.max(1)[1]
                    meter_test.update('acc', pred.eq(labels.data.view_as(pred)).sum().item())

            recorder.logger.info(f'Test Avg. Loss: {meter_test.loss.avg:.6f}, '
                                 f'accuracy: {100 * meter_test.acc.sum / test_num:.2f}%')
            recorder.add_scalars_from_dict({'loss_test': meter_test.loss.avg,
                                            'accuracy': meter_test.acc.sum / test_num},
                                           global_step=epoch)

        if args.ckp_freq > 0 and epoch % args.ckp_freq == args.ckp_freq - 1:
            recorder.save_model({'state': net.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scheduler': scheduler.state_dict()},
                                f'epoch{epoch}-ckp.pt')

    recorder.save_model({'state': net.state_dict(), 'acc': 100 * meter_test.acc.sum / test_num},
                        f'{args.dataset}-{args.model}.pt')
