# 2019.07.24-Changed output of forward function
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class SimpleMultiHeadResNet(nn.Module):
    def __init__(self, block, num_blocks, headers, num_classes=10):
        super(SimpleMultiHeadResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4_1 = deepcopy(layer4[:-1])
        self.layer4_2 = nn.ModuleList([deepcopy(layer4[-1]) for _ in range(headers)])
        self.linear = nn.ModuleList([nn.Linear(512 * block.expansion, num_classes) for _ in range(headers)])
        del layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = self.layer3(out)  # 8x8
        out = self.layer4_1(out)  # 4x4

        out = [layer4_2_i(out) for layer4_2_i in self.layer4_2]
        out = [F.avg_pool2d(out_i, 4) for out_i in out]
        out = [out_i.view(out_i.size(0), -1) for out_i in out]
        out = [linear_i(out_i) for linear_i, out_i in zip(self.linear, out)]
        return out


def smhresnet18(headers, num_classes):
    return SimpleMultiHeadResNet(BasicBlock, [2, 2, 2, 2], headers, num_classes)


def smhresnet34(headers, num_classes):
    return SimpleMultiHeadResNet(BasicBlock, [3, 4, 6, 3], headers, num_classes)


def smhresnet50(headers, num_classes):
    return SimpleMultiHeadResNet(Bottleneck, [3, 4, 6, 3], headers, num_classes)


def smhresnet101(headers, num_classes):
    return SimpleMultiHeadResNet(Bottleneck, [3, 4, 23, 3], headers, num_classes)


def smhresnet152(headers, num_classes):
    return SimpleMultiHeadResNet(Bottleneck, [3, 8, 36, 3], headers, num_classes)
