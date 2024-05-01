import torch
import torch.nn as nn
import math
from typing import List
from utils.anchor_utils import AnchorGenerator
from torchvision.ops.boxes import box_convert


class FCOSRegressionHead(nn.Module):
    def __init__(self, in_channels):
        super(FCOSRegressionHead, self).__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.bbox_reg, self.bbox_ctrness]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        bbox_feature = self.conv(x)

        bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))

        bbox_ctrness = self.bbox_ctrness(bbox_feature)

        return bbox_regression, bbox_ctrness


class FCOSClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCOSClassificationHead, self).__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        cls_logits = self.conv(x)
        cls_logits = self.cls_logits(cls_logits)
        return cls_logits


class FCOSHead(nn.Module):
    def __init__(self, in_channels_list: List, anchors, aspect_ratios, num_classes):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        self.nl = len(in_channels_list)
        self.na = len(anchors[0])
        self.anchors = AnchorGenerator(anchors, aspect_ratios)

        self.head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.head.append(
                nn.ModuleList([FCOSClassificationHead(in_channels, num_classes), FCOSRegressionHead(in_channels)])
            )

    def forward(self, x, H, W):
        z = []
        device = x[0].device
        imgsze = torch.tensor([W, H], device=device)
        anchors, strides = self.anchors(imgsze, x)

        for i in range(self.nl):
            classification_head = self.head[i][0]
            regression_head = self.head[i][1]
            cls_logits = classification_head(x[i])
            bbox_regression, bbox_ctrness = regression_head(x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = torch.cat([bbox_regression, bbox_ctrness, cls_logits], 1).permute([0, 2, 3, 1]).contiguous()
            if not self.training:  # inference
                shape = 1, self.na, ny, nx, 2  # grid shape

                anchor = anchors[i].view((1, 1, ny, nx, 2)).expand(shape)

                pxy = 0.5 * (anchor[..., :2] + anchor[..., 2:])

                pwh = anchor[..., 2:] - anchor[..., :2]

                x1y1, x2y2, conf, cls = x[i].split((2, 2, 1, self.num_classes), -1)
                x1y1 = pxy - (x1y1 * pwh)
                x2y2 = pxy + (x2y2 * pwh)
                xywh = box_convert(torch.cat([x1y1, x2y2], -1), in_fmt='xyxy', out_fmt='cxcywh')

                conf, cls = conf.sigmoid(), cls.sigmoid()

                cls_scores, _ = torch.max(cls, dim=-1)

                conf = (conf * cls_scores).sqrt()

                y = torch.cat((xywh, conf, cls), -1)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)
