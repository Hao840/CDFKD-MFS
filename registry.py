from nets import gan, resnet, resnet8x, mhresnet, mhresnet8x, smhresnet, smhresnet8x, wrn4x, mhwrn4x

__all__ = ['datainfo', 'models']

models = {
    # for 32x32 images
    'resnet8x18': resnet8x.resnet18,
    'resnet8x34': resnet8x.resnet34,
    'mhresnet8x18': mhresnet8x.mhresnet18,
    'mhresnet8x34': mhresnet8x.mhresnet34,
    'smhresnet8x18': smhresnet8x.smhresnet18,
    'smhresnet8x34': smhresnet8x.smhresnet34,
    'wrn4x161': wrn4x.wrn4x161,
    'wrn4x162': wrn4x.wrn4x162,
    'wrn4x402': wrn4x.wrn4x402,
    'mhwrn4x161': mhwrn4x.mhwrn4x161,
    'mhwrn4x162': mhwrn4x.mhwrn4x162,
    'mhwrn4x402': mhwrn4x.mhwrn4x402,

    # for larger images
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'mhresnet18': mhresnet.mhresnet18,
    'mhresnet34': mhresnet.mhresnet34,
    'smhresnet18': smhresnet.smhresnet18,
    'smhresnet34': smhresnet.smhresnet34,
}

datainfo = {
    'cifar100': {'generator':gan.Generator,
                 'num_classes': 100,
                 'img_size': 32,
                 'train_num': 50000,
                 'mean': (0.5071, 0.4865, 0.4409),
                 'std':(0.2673, 0.2564, 0.2762)
                 },
    'caltech101': {'generator':gan.GeneratorLarge,
                   'num_classes': 101,
                   'img_size': 128,
                   'train_num': 6983,
                   # 'mean': (0.5453, 0.5283, 0.5022),
                   # 'std': (0.2422, 0.2392, 0.2406)
                   'mean': (0.5, 0.5, 0.5),
                   'std': (0.5, 0.5, 0.5)
                   },
    'miniimagenet': {'generator':gan.GeneratorLargeBN,
                     'num_classes': 100,
                     'img_size': 224,
                     'train_num': 48000,
                     'mean': (0.485, 0.456, 0.406),
                     'std': (0.229, 0.224, 0.225)
                     }
}
