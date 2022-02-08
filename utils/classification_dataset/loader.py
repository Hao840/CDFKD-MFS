from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .caltech import Caltech101
from .config import cfg
from .imagenet import ImageNet
from .mini_imagenet import MiniImagenet


def get_loader(dataset, batch_size, num_workers=4, transform_train=None, transform_test=None):
    root = cfg.default_root
    persistent_workers = True if num_workers > 0 else False

    if dataset == 'cifar10':
        Dataset = CIFAR10

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    elif dataset == 'cifar100':
        Dataset = CIFAR100

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

    elif dataset == 'caltech101':
        Dataset = Caltech101

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
                # transforms.Normalize((0.5453, 0.5283, 0.5022), (0.2422, 0.2392, 0.2406))
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
                # transforms.Normalize((0.5453, 0.5283, 0.5022), (0.2422, 0.2392, 0.2406))
            ])

    elif dataset == 'miniimagenet':
        Dataset = MiniImagenet

        if transform_train == None:
            transform_train = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    elif dataset == 'imagenet':
        Dataset = ImageNet

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    else:
        raise NotImplementedError

    data_train = Dataset(root=root,
                         transform=transform_train)
    data_test = Dataset(root=root,
                        train=False,
                        transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   persistent_workers=persistent_workers)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers,
                                  persistent_workers=persistent_workers)

    return data_train_loader, data_test_loader
