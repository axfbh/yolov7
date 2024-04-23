import torch.nn as nn

from typing import List
import torch
import math


class YoloHead(nn.Module):
    def __init__(self, out_channle_list: List, num_anchors, num_classes):
        super(YoloHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.head = nn.ModuleList([nn.Conv2d(out_channle_list[0], num_anchors * num_classes, 1, 1, 0),
                                   nn.Conv2d(out_channle_list[1], num_anchors * num_classes, 1, 1, 0),
                                   nn.Conv2d(out_channle_list[2], num_anchors * num_classes, 1, 1, 0)])

        self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        for layer, s in zip(self.head, stride):
            if isinstance(layer, nn.Conv2d):
                b = layer.bias.view(self.num_anchors, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                b.data[:, 5:self.num_classes] += math.log(0.6 / (self.num_classes - 5 - 0.99999))
                layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: List):
        p5 = self.head[0](x[0])
        p6 = self.head[1](x[1])
        p7 = self.head[2](x[2])

        return p5, p6, p7
