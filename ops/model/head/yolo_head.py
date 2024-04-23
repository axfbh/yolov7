import torch.nn as nn

from typing import List
import torch
import math


class YoloHead(nn.Module):
    def __init__(self, out_channle_list: List, num_anchors, num_classes):
        super(YoloHead, self).__init__()

        self.head_p5 = nn.Conv2d(out_channle_list[0], num_anchors * num_classes, 1, 1, 0)

        self.head_p6 = nn.Conv2d(out_channle_list[1], num_anchors * num_classes, 1, 1, 0)

        self.head_p7 = nn.Conv2d(out_channle_list[2], num_anchors * num_classes, 1, 1, 0)

        self.reset_parameters()

    def reset_parameters(self):
        self.head_p5.bias.data[4] += math.log(8 / (640 / 8) ** 2)
        self.head_p5.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

        self.head_p6.bias.data[4] += math.log(8 / (640 / 16) ** 2)
        self.head_p6.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

        self.head_p7.bias.data[4] += math.log(8 / (640 / 32) ** 2)
        self.head_p7.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

    def forward(self, x: List):
        p5 = self.head_p5(x[0])
        p6 = self.head_p6(x[1])
        p7 = self.head_p7(x[2])

        return p5, p6, p7
