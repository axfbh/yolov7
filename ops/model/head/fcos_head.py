import torch
import torch.nn as nn
import math


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
    def __init__(self, in_channels, num_classes):
        super(FCOSHead, self).__init__()
        self.classification_head = FCOSClassificationHead(in_channels, num_classes)
        self.regression_head = FCOSRegressionHead(in_channels)

    def forward(self, x):
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)

        return bbox_regression, bbox_ctrness, cls_logits
