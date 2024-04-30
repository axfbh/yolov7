import torch.nn as nn
import torch
from ops.model.neck.spp import SPPCSPC
from ops.model.head.yolo_head import YoloV7Head
from ops.model.misc.rep_conv import RepConv2d
from ops.model.backbone.elandarknet import ElanDarkNet, CBS, MP1, Elan
from ops.model.backbone.utils import _elandarknet_extractor
from typing import List


class YoloV7(nn.Module):
    def __init__(self, anchors: List, num_classes: int, phi: str):
        super(YoloV7, self).__init__()

        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        panet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]

        self.backbone = _elandarknet_extractor(ElanDarkNet(transition_channels, block_channels, n, phi), 5)

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16,
                               conv_layer=CBS,
                               activation_layer=nn.SiLU)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_P5 = CBS(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = CBS(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Elan(transition_channels * 16, panet_channels * 4,
                                        transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4 = CBS(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = CBS(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Elan(transition_channels * 8, panet_channels * 2,
                                        transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1 = MP1(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Elan(transition_channels * 16, panet_channels * 4,
                                          transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2 = MP1(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Elan(transition_channels * 32, panet_channels * 8,
                                          transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_1 = CBS(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = CBS(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = CBS(transition_channels * 16, transition_channels * 32, 3, 1)

        self.head = YoloV7Head([transition_channels * 8, transition_channels * 16, transition_channels * 32],
                               anchors,
                               num_classes)

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backbone(x)

        feat1, feat2, feat3 = x['0'], x['1'], x['2']

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        return self.head([P3, P4, P5], H, W)
