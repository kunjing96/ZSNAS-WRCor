from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from .nasbench101_ops import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class NASBench101Network(nn.Module):
    def __init__(self, spec, stem_out, num_stacks, num_mods, num_classes, bn=True):
        super(NASBench101Network, self).__init__()

        self.spec=spec
        self.stem_out=stem_out 
        self.num_stacks=num_stacks 
        self.num_mods=num_mods
        self.num_classes=num_classes

        self.layers = nn.ModuleList([])

        in_channels = 3
        out_channels = stem_out

        # initial stem convolution
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1, bn=bn)
        self.layers.append(stem_conv)

        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for _ in range(num_mods):
                cell = Cell(spec, in_channels, out_channels, bn=bn)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x, pre_GAP=False):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        out = torch.mean(x, (2, 3))
        out = self.classifier(out)

        if pre_GAP:
            return x
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
