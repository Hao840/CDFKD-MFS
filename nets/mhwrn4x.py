import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import wrn4x


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class MultiHeadWRN(nn.Module):
    def __init__(self, headers, num_classes, backbone, ch):
        super(MultiHeadWRN, self).__init__()

        backbone = backbone(num_classes=num_classes)

        self.shared_conv = backbone.conv1

        self.shared_layer1 = backbone.block1
        self.shared_layer2 = backbone.block2
        self.shared_layer3 = backbone.block3

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder1 = nn.ModuleList([SepConv(ch[0], ch[1]) for _ in range(headers)])
        self.encoder2 = nn.ModuleList([SepConv(2 * ch[1], ch[2]) for _ in range(headers)])
        self.encoder3 = nn.ModuleList([SepConv(2 * ch[2], ch[2]) for _ in range(headers)])

        self.linear = nn.ModuleList([nn.Linear(ch[2], num_classes) for _ in range(headers)])

    def forward(self, x, out_feature=False):
        # Shared convolution
        x = self.shared_conv(x)

        # Shared block
        u_1 = self.shared_layer1(x)
        u_2 = self.shared_layer2(u_1)
        u_3 = self.shared_layer3(u_2)

        a_1 = [encoder1_i(u_1) for encoder1_i in self.encoder1]
        a_2 = [encoder2_i(torch.cat((u_2, a_1_i), dim=1)) for encoder2_i, a_1_i in zip(self.encoder2, a_1)]
        a_3 = [encoder3_i(torch.cat((u_3, a_2_i), dim=1)) for encoder3_i, a_2_i in zip(self.encoder3, a_2)]

        out = [F.avg_pool2d(a_3_i, 4) for a_3_i in a_3]
        feat = [(lambda x: x.view(x.size(0), -1))(out_i) for out_i in out]
        out = [linear_i(feat_i) for linear_i, feat_i in zip(self.linear, feat)]
        if out_feature == False:
            return out
        else:
            return out, feat


def mhwrn4x161(headers, num_classes):
    return MultiHeadWRN(headers, num_classes, backbone=wrn4x.wrn4x161, ch=[16, 32, 64])


def mhwrn4x162(headers, num_classes):
    return MultiHeadWRN(headers, num_classes, backbone=wrn4x.wrn4x162, ch=[32, 64, 128])


def mhwrn4x402(headers, num_classes):
    return MultiHeadWRN(headers, num_classes, backbone=wrn4x.wrn4x402, ch=[32, 64, 128])
