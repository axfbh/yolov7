import torch.nn as nn
from ops.model.misc.elan_conv import ElanH
from ops.model.neck.spp import SPPCSPC
from ops.model.neck.panet import PaNet
from ops.model.head.yolo_head import YoloHead
from ops.model.misc.rep_conv import RepConv2d
from ops.model.backbone.elandarknet53 import ElanDarkNet53
from ops.model.backbone.utils import _elandarknet_extractor

#
# class Yolov7(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#
#         self.backbone = _elandarknet_extractor(ElanDarkNet53(), 5)
#
#         self.sppcspc = SPPCSPC(1024, 512, activation_layer=nn.SiLU)
#
#         self.neck = PaNet([512, 256, 128], ElanH, activation_layer=nn.SiLU)
#
#         self.cv1 = RepConv2d(512, 1024)
#         self.cv2 = RepConv2d(256, 512)
#         self.cv3 = RepConv2d(128, 256)
#
#         self.head = YoloHead([256, 512, 1024], num_classes)
#
#     def forward(self, x):
#         x = self.backbone(x)
#
#         p7 = self.sppcspc(x['2'])
#
#         sample = [p7, x['1'], x['0']]
#
#         p5, p6, p7 = self.neck(sample)
#
#         p5 = self.cv3(p5)
#         p6 = self.cv2(p6)
#         p7 = self.cv1(p7)
#
#         head = self.head([p5, p6, p7])
#
#         return head


import torch
import torch.nn as nn
import math


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = Focus(3, base_channels, k=3)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


class YoloBody(nn.Module):
    def __init__(self, anchors_num, num_classes, phi):
        super(YoloBody, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80,80,256
        #   40,40,512
        #   20,20,1024
        # ---------------------------------------------------#
        self.backbone = CSPDarknet(base_channels, base_depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, anchors_num * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, anchors_num * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, anchors_num * (5 + num_classes), 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.yolo_head_P3.bias.data[4] += math.log(8 / (640 / 8) ** 2)
        self.yolo_head_P3.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

        self.yolo_head_P4.bias.data[4] += math.log(8 / (640 / 16) ** 2)
        self.yolo_head_P4.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

        self.yolo_head_P5.bias.data[4] += math.log(8 / (640 / 32) ** 2)
        self.yolo_head_P5.bias.data[5:] += math.log(0.6 / (20 - 0.99999))

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)

        P5 = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4 = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out2, out1, out0


def get_model(args):
    return YoloBody(anchors_num=3, num_classes=args.num_classes, phi='s')
