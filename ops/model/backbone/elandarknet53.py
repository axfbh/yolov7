import torch
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial
import torch.nn as nn
from ops.model.misc.elan_conv import Elan

CBS = partial(Conv2dNormActivation, bias=False, inplace=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)


class MP1(nn.Module):
    def __init__(self, in_ch, expand_ratio=0.5):
        super(MP1, self).__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.cv1 = CBS(in_ch, mid_ch, 1)
        self.cv2 = nn.Sequential(CBS(in_ch, mid_ch, 1),
                                 CBS(mid_ch, mid_ch, 3, 2))

    def forward(self, x):
        x1 = self.maxpool(x)
        x1 = self.cv1(x1)

        x2 = self.cv2(x)

        x = torch.cat([x1, x2], dim=1)
        return x


class ElanDarkNet53(nn.Module):
    def __init__(self, num_classes=1000):
        super(ElanDarkNet53, self).__init__()

        self.stem = nn.Sequential(CBS(3, 32, 3),
                                  CBS(32, 64, 3, 2),
                                  CBS(64, 64, 3))

        self.stage1 = nn.Sequential(CBS(64, 128, 3, 2),
                                    Elan(128, 256))

        self.stage2 = nn.Sequential(MP1(256),
                                    Elan(256, 512))

        self.stage3 = nn.Sequential(MP1(512),
                                    Elan(512, 1024))

        self.stage4 = nn.Sequential(MP1(1024),
                                    Elan(1024, 1024))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
