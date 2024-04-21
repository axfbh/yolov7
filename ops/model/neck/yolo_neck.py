import torch.nn as nn

from ops.model.neck.spp import SPP

from ops.model.neck.panet import PaNet

from typing import List


class ConvolutionalLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None):
        p = (k - 1) // 2 if p is None else p
        super(ConvolutionalLayer, self).__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
        )


class YVBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(YVBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_ch, out_ch, 1),
            ConvolutionalLayer(out_ch, in_ch, 3),
            ConvolutionalLayer(in_ch, out_ch, 1),
            ConvolutionalLayer(out_ch, in_ch, 3),
            ConvolutionalLayer(in_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.conv(x)


class YoloV4Neck(nn.Module):
    def __init__(self):
        super(YoloV4Neck, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(1024, 512, 1),
            ConvolutionalLayer(512, 1024, 3),
            ConvolutionalLayer(1024, 512, 1),
            SPP([5, 9, 13]),
            ConvolutionalLayer(2048, 512, 1),
            ConvolutionalLayer(512, 1024, 3),
            ConvolutionalLayer(1024, 512, 1),
        )

        self.panet = PaNet([512, 256, 128], YVBlock, use_cat=True)

    def forward(self, x: List):
        x[0] = self.conv(x[0])
        return self.panet(x)
