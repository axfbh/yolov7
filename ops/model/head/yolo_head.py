import torch.nn as nn

from typing import List


# 定义卷积层：经过这个层仅变换通道数
class ConvolutionalLayer(nn.Sequential):
    def __init__(self, out_channels, kernel_size, stride=1, padding=None):
        padding = (kernel_size - 1) // 2 if padding is None else padding
        super(ConvolutionalLayer, self).__init__(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )


class YoloHead(nn.Module):
    def __init__(self, out_channle_list: List, num_classes):
        super(YoloHead, self).__init__()
        self.head_p5 = nn.Sequential(
            ConvolutionalLayer(out_channle_list[0], 3),
            nn.Conv2d(out_channle_list[0], num_classes, 1, 1, 0, bias=False)
        )

        self.head_p6 = nn.Sequential(
            ConvolutionalLayer(out_channle_list[1], 3),
            nn.Conv2d(out_channle_list[1], num_classes, 1, 1, 0, bias=False)
        )

        self.head_p7 = nn.Sequential(
            ConvolutionalLayer(out_channle_list[2], 3),
            nn.Conv2d(out_channle_list[2], num_classes, 1, 1, 0, bias=False)
        )

    def forward(self, x: List):
        p5 = self.head_p5(x[0])
        p6 = self.head_p6(x[1])
        p7 = self.head_p7(x[2])

        return p5, p6, p7
