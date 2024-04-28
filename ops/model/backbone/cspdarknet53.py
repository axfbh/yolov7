import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation

BN = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
CBM = partial(Conv2dNormActivation, bias=False, inplace=True, norm_layer=BN, activation_layer=nn.Mish)


class DownSampleLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(DownSampleLayer, self).__init__(
            CBM(in_ch, out_ch, 3, 2)
        )


class ResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualLayer, self).__init__()
        self.conv = nn.Sequential(
            CBM(in_ch, out_ch, 1),
            nn.Conv2d(out_ch, in_ch, 3, 1, 1, bias=False)
        )

        self.bn = nn.BatchNorm2d(in_ch)

        self.mish = nn.Mish()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) + x
        return self.mish(out)


class WrapLayer(nn.Module):
    def __init__(self, in_ch, count=1, first=False):
        super(WrapLayer, self).__init__()
        out_ch = in_ch if first else in_ch // 2
        self.trans_0 = CBM(in_ch, out_ch, 1)

        self.trans_1 = CBM(in_ch, out_ch, 1)

        self.trans_cat = CBM(in_ch * 2 if first else in_ch, in_ch, 1)

        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(ResidualLayer(out_ch, in_ch // 2))

        self.conv1 = CBM(out_ch, out_ch, 1)

    def forward(self, x):
        # ----------- 两分支 -----------
        out0 = self.trans_0(x)
        out1 = self.trans_1(x)

        for conv in self.make_layers:
            out0 = conv(out0)

        out0 = self.conv1(out0)
        out = torch.cat([out0, out1], 1)
        out = self.trans_cat(out)
        return out


class CSPDarkNet53(nn.Module):
    def __init__(self, num_classes=100):
        super(CSPDarkNet53, self).__init__()

        self.stem = nn.Sequential(
            CBM(3, 32, 3),
            DownSampleLayer(32, 64),
            WrapLayer(64, 1, first=True),
        )

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(64, 128),
            WrapLayer(128, 2),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(128, 256),
            WrapLayer(256, 8),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(256, 512),
            WrapLayer(512, 8),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(512, 1024),
            WrapLayer(1024, 4),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def cspdarknet53(num_classes) -> CSPDarkNet53:
    """ Constructs a ResNet-50 model.
    """

    return CSPDarkNet53(num_classes)
