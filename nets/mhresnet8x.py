import torch
import torch.nn as nn

from nets import resnet8x


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


class MultiHeadResNet(nn.Module):
    def __init__(self, headers, num_classes, backbone=resnet8x.resnet18):
        super(MultiHeadResNet, self).__init__()

        backbone = backbone(num_classes=num_classes)

        ch = [64, 128, 256, 512]

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, nn.ReLU(inplace=True))

        self.shared_layer1_b = backbone.layer1[:-1]
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder1 = nn.ModuleList([SepConv(ch[0], ch[1]) for _ in range(headers)])
        self.encoder2 = nn.ModuleList([SepConv(2 * ch[1], ch[2]) for _ in range(headers)])
        self.encoder3 = nn.ModuleList([SepConv(2 * ch[2], ch[3]) for _ in range(headers)])
        self.encoder4 = nn.ModuleList([SepConv(2 * ch[3], ch[3]) for _ in range(headers)])

        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(headers)])

    def forward(self, x, out_feature=False):
        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)  # 64x32x32

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)  # 128x16x16

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)  # 256x8x8

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)  # 512x4x4

        a_1 = [encoder1_i(u_1_t) for encoder1_i in self.encoder1]
        a_2 = [encoder2_i(torch.cat((u_2_t, a_1_i), dim=1)) for encoder2_i, a_1_i in zip(self.encoder2, a_1)]
        a_3 = [encoder3_i(torch.cat((u_3_t, a_2_i), dim=1)) for encoder3_i, a_2_i in zip(self.encoder3, a_2)]
        a_4 = [encoder4_i(torch.cat((u_4_t, a_3_i), dim=1)) for encoder4_i, a_3_i in zip(self.encoder4, a_3)]

        out = [self.adaptive_avg_pool2d(a_4_i) for a_4_i in a_4]
        feat = [(lambda x: x.view(x.size(0), -1))(out_i) for out_i in out]
        out = [linear_i(feat_i) for linear_i, feat_i in zip(self.linear, feat)]
        if out_feature == False:
            return out
        else:
            return out, feat


def mhresnet18(headers, num_classes):
    return MultiHeadResNet(headers, num_classes, backbone=resnet8x.resnet18)

def mhresnet34(headers, num_classes):
    return MultiHeadResNet(headers, num_classes, backbone=resnet8x.resnet34)


if __name__ == '__main__':
    net = MultiHeadResNet(3, 10)
    img = torch.randn(1, 3, 32, 32)
    out = net(img)
