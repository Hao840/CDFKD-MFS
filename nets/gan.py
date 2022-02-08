import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Generator(nn.Module):
    def __init__(self, nz=256, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.out_block = nn.Sequential(
            nn.Conv2d(64, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = self.out_block(img)
        return img


class GeneratorLarge(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """

    def __init__(self, nz=256, img_size=128, nc=3):
        super(GeneratorLarge, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, 64 * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(64 * 8),

            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 2x

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x

            nn.Conv2d(64, nc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class GeneratorLargeBN(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """

    def __init__(self, nz=256, img_size=128, nc=3):
        super(GeneratorLargeBN, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, 64 * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(64 * 8),

            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 2x

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x

            nn.Conv2d(64, nc, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

if __name__ == "__main__":
    import torch

    net = GeneratorLarge(1000, 224, 3)
    z = torch.randn(1, 1000)
    imgs = net(z)
