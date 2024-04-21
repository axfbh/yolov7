import torch.nn as nn

import torch
from typing import List
from functools import partial


class ConvolutionalLayer(nn.Sequential):
    def __init__(self, out_ch, k, s=1, p=None, activation_layer=None):
        activation_layer = nn.LeakyReLU if activation_layer is None else activation_layer
        p = (k - 1) // 2 if p is None else p
        super(ConvolutionalLayer, self).__init__(
            nn.LazyConv2d(out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer(),
        )


class AddBlock(nn.Module):
    def __init__(self, out_ch, conv_layer):
        super(AddBlock, self).__init__()
        self.conv = conv_layer(out_ch, out_ch)

    def forward(self, x1, x2):
        x = x1 + x2
        return self.conv(x)


class CatBlock(nn.Module):
    def __init__(self, out_ch, conv_layer):
        super(CatBlock, self).__init__()
        self.conv = conv_layer(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class MixDownLayer(nn.Module):
    def __init__(self, out_ch, block, activation_layer):
        super(MixDownLayer, self).__init__()

        self.down = ConvolutionalLayer(out_ch, 3, 2, activation_layer=activation_layer)

        self.conv = block(out_ch)

    def forward(self, x1, x2):
        x1 = self.down(x1)
        return self.conv(x1, x2)


class MixUpLayer(nn.Module):
    def __init__(self, out_ch, block, activation_layer):
        super(MixUpLayer, self).__init__()

        self.upsample = nn.Sequential(
            ConvolutionalLayer(out_ch, 1, activation_layer=activation_layer),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.conv1 = ConvolutionalLayer(out_ch, 1, activation_layer=activation_layer)

        self.conv2 = block(out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x2 = self.conv1(x2)
        return self.conv2(x1, x2)


class PanNetTopDown(nn.Module):
    def __init__(self, out_channels_list, conv_layer, use_cat, activation_layer=None):
        super(PanNetTopDown, self).__init__()

        block = partial(CatBlock, conv_layer=conv_layer) if use_cat else partial(AddBlock, conv_layer=conv_layer)

        self.mix_up = nn.Sequential()
        for i in range(len(out_channels_list)):
            out_ch = out_channels_list[i]
            self.mix_up.append(MixUpLayer(out_ch, block, activation_layer))

    def forward(self, x: List):
        up_sampling = [x.pop(0)]

        for i in range(len(x)):
            x1 = x[i]
            conv = self.mix_up[i]
            up_sampling.append(conv(up_sampling[-1], x1))

        return up_sampling


class PanNetBottomUp(nn.Module):
    def __init__(self, out_channels_list, conv_layer, use_cat, activation_layer=None):
        super(PanNetBottomUp, self).__init__()

        block = partial(CatBlock, conv_layer=conv_layer) if use_cat else partial(AddBlock, conv_layer=conv_layer)

        self.mix_down = nn.Sequential()
        for i in range(len(out_channels_list)):
            out_ch = out_channels_list[i]
            self.mix_down.append(MixDownLayer(out_ch, block, activation_layer))

    def forward(self, x: List):
        down_sampling = [x.pop(0)]

        for i in range(len(x)):
            x1 = x[i]
            conv = self.mix_down[i]
            down_sampling.append(conv(down_sampling[-1], x1))

        return down_sampling


class PaNet(nn.Module):
    def __init__(self,
                 out_channels_list,
                 conv_layer,
                 use_cat=True,
                 activation_layer=None):
        super(PaNet, self).__init__()

        self.up = PanNetTopDown(out_channels_list[1:], conv_layer, use_cat, activation_layer)

        out_channels_list = out_channels_list[::-1]

        self.down = PanNetBottomUp(out_channels_list[1:], conv_layer, use_cat, activation_layer)

    def forward(self, x: List):
        up_sampling = self.up(x)
        up_sampling = up_sampling[::-1]
        down_sampling = self.down(up_sampling)
        return down_sampling
