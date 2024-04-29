import torch.nn as nn
import torch
from ops.model.backbone.cspdarknet53 import CSPDarkNet53, CBM
from ops.model.backbone.utils import _cspdarknet_extractor
from ops.model.head.yolo_head import YoloHead
from ops.model.neck.spp import SPP


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CBM(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        CBM(in_filters, filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
    )
    return m


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        CBM(in_filters, filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
    )
    return m


class YoloV4(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloV4, self).__init__()

        self.backbone = _cspdarknet_extractor(CSPDarkNet53(), 5)

        self.cov1 = make_three_conv([512, 1024], 1024)
        self.spp = SPP([5, 9, 13])
        self.cov2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = CBM(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = CBM(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.down_sample1 = CBM(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.down_sample2 = CBM(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        self.head = YoloHead([128, 256, 512], num_anchors, num_classes)

    def forward(self, x):
        x = self.backbone(x)

        feat1, feat2, feat3 = x['0'], x['1'], x['2']

        P5 = self.cov1(feat3)
        P5 = self.spp(P5)
        P5 = self.cov2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(feat2)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(feat1)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        P5 = self.make_five_conv4(P5)

        head = self.head([P3, P4, P5])

        return head


def get_model(args):
    model = YoloV4(3, args.num_classes)
    return model
