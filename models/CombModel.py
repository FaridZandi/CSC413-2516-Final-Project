import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch import Tensor

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import warnings

from typing import Callable, Any, Optional, Tuple, List
from collections import namedtuple

configures = {
    1: [("B", 64), ("B", 64),
        ("M", 2),
        ("B", 128), ("B", 128), ("B", 128),
        ("M", 2),
        ("B", 256), ("B", 256), ("B", 256),
        ("M", 2)],  # 45 percent
    2: [("B", 64), ("R", 64),
        ("M", 2),
        ("B", 128), ("R", 128), ("R", 128),
        ("M", 2),
        ("B", 256), ("R", 256), ("R", 256),
        ("M", 2)],
    3: [("B", 64), ("B", 64),
        ("M", 2),
        ("I", 128), ("I", 128), ("I", 128),
        ("M", 2),
        ("I", 256), ("I", 256), ("I", 256),
        ("M", 2)],
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False)


class BasicConv2d(nn.Module):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class ResNetBasicBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
    ) -> None:
        super(ResNetBasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class InceptionA(nn.Module):

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
    ) -> None:
        super(InceptionA, self).__init__()
        conv_block = BasicConv2d

        path_out_planes = out_planes / 4
        self.branch1x1 = conv_block(in_planes, path_out_planes, kernel_size=1)

        self.branch5x5_1 = conv_block(in_planes, path_out_planes, kernel_size=1)
        self.branch5x5_2 = conv_block(path_out_planes, path_out_planes, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_planes, path_out_planes, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(path_out_planes, path_out_planes, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(path_out_planes, path_out_planes, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_planes, path_out_planes, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class CombModel(nn.Module):
    def __init__(self, num_classes, configuration):
        super(CombModel, self).__init__()

        self.features = make_layers(configures[configuration], True)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(configure, batch_norm):
    layers = []
    in_channels = 3
    for type, param in configure:
        print(type, param)
        if type == 'M':
            layers += [nn.MaxPool2d(kernel_size=param, stride=param)]
        elif type == "B":
            layers += [BasicBlock(in_channels, param)]
            in_channels = param
        elif type == "R":
            layers += [ResNetBasicBlock(in_channels, param)]
            in_channels = param
        elif type == "I":
            layers += [InceptionA(in_channels, param)]
            in_channels = param
    return nn.Sequential(*layers)
