'''
Dreaming to Distill Data-free Knowledge Transfer via DeepInversion
step 2. implement vanilla KD (KD.py) with samples generated in step 1.'''
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from registry import *
from utils import Recorder, MultiMeter, kdloss
from utils.classification_dataset import get_loader


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, size=None):
        self.root = os.path.abspath(root)
        self.images = self._collect_all_images(self.root)
        if size is not None:
            self.images = self.images[:size]
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='DI-default')
    parser.add_argument('--data_root', type=str, default='data/DIpre')

    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--model_t', type=str, default='resnet8x34')
    parser.add_argument('--model_s', type=str, default='resnet8x18')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bsz', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_milestone', type=int, nargs='+', default=[80, 120])
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--id_t', type=int, default=1)

    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--ckp_freq', type=int, default=-1)

    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    recorder = Recorder(base_path='result/main/DI',
                        exp_name=args.exp_name,
                        logger_name=__name__,
                        code_path=__file__)
    recorder.logger.info(args)

    num_classes = datainfo[args.dataset]['num_classes']
    train_num = datainfo[args.dataset]['train_num']
    teacher = models[args.model_t](num_classes=num_classes).cuda()
    state_dict = torch.load(f'ckp/{args.dataset}-{args.model_t}-{args.id_t}.pt')['state']
    teacher.load_state_dict(state_dict)
    teacher.eval()

    student = models[args.model_s](num_classes=num_classes).cuda()

    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestone, 0.1)

    _train_loader, test_loader = get_loader(args.dataset, args.bsz, args.num_workers)
    train_set = UnlabeledImageDataset(root=os.path.join(args.data_root, args.dataset),
                                      transform=_train_loader.dataset.transform,
                                      size=train_num)
    train_loader = DataLoader(train_set, batch_size=args.bsz, shuffle=True, num_workers=args.num_workers,
                              persistent_workers=True if args.num_workers > 0 else False)

    train_num = len(train_loader.dataset)
    test_num = len(test_loader.dataset)

    meter_train = MultiMeter()
    meter_train.register(['loss'])
    meter_test = MultiMeter()
    meter_test.register(['loss', 'acc'])

    for epoch in range(args.epochs):
        meter_train.reset()
        student.train()
        for images in tqdm(train_loader, leave=False, unit_scale=True, desc=f'Epoch-{epoch} train'):
            images = images.cuda()
            with torch.no_grad():
                target = teacher(images)
            output = student(images)
            loss = kdloss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter_train.update('loss', loss.item(), len(images))

        recorder.logger.info(f'Train - Epoch {epoch}, Loss: {meter_train.loss.avg:.6f}')
        recorder.writer.add_scalar('loss_train', meter_train.loss.avg, epoch)

        scheduler.step()

        if epoch % args.test_freq == args.test_freq - 1 or epoch >= args.epochs - 10:
            meter_test.reset()
            student.eval()
            with torch.no_grad():
                for images, labels in tqdm(test_loader, leave=False, unit_scale=True, desc=f'Epoch-{epoch} test'):
                    images, labels = images.cuda(), labels.cuda()
                    output = student(images)

                    meter_test.update('loss', F.cross_entropy(output, labels).item(), n=len(images))
                    pred = output.data.max(1)[1]
                    meter_test.update('acc', pred.eq(labels.data.view_as(pred)).sum().item())

            recorder.logger.info(f'Test Avg. Loss: {meter_test.loss.avg:.6f}, '
                                 f'accuracy: {100 * meter_test.acc.sum / test_num:.2f}%')
            recorder.add_scalars_from_dict({'loss_test': meter_test.loss.avg,
                                            'accuracy': meter_test.acc.sum / test_num},
                                           global_step=epoch)

        if args.ckp_freq > 0 and epoch % args.ckp_freq == args.ckp_freq - 1:
            recorder.save_model({'state': student.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scheduler': scheduler.state_dict()},
                                f'epoch{epoch}-ckp.pt')

    recorder.save_model({'state': student.state_dict(), 'acc': 100 * meter_test.acc.sum / test_num},
                        f'{args.dataset}-{args.model_s}.pt')
