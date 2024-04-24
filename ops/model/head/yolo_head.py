import torch.nn as nn

from typing import List
import torch
import math
from ops.detection.anchor_utils import make_grid


class YoloV7Head(nn.Module):
    def __init__(self, out_channle_list: List, anchors: List, num_classes: int):
        super(YoloV7Head, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.num_classes = num_classes
        self.no = num_classes + 5
        self.head = nn.ModuleList([nn.Conv2d(out_channle_list[0], self.na * self.no, 1, 1, 0),
                                   nn.Conv2d(out_channle_list[1], self.na * self.no, 1, 1, 0),
                                   nn.Conv2d(out_channle_list[2], self.na * self.no, 1, 1, 0)])

        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        # self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        for layer, s in zip(self.head, stride):
            if isinstance(layer, nn.Conv2d):
                b = layer.bias.view(self.num_anchors, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                b.data[:, 5:self.num_classes] += math.log(0.6 / (self.num_classes - 0.99999))
                layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: List, H, W):
        z = []  # inference output
        device = self.anchors.device
        for i in range(self.na):
            x[i] = self.head[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                anchor = self.anchors[i] / torch.tensor([W, H], device=device) * torch.tensor([nx, ny], device=device)

                stride = torch.tensor([W, H], device=device) / torch.tensor([nx, ny], device=device)

                shape = 1, self.na, ny, nx, 2  # grid shape
                grid = make_grid(ny, nx, 1, 1, self.anchors.dtype, device).view((1, 1, ny, nx, 2)).expand(shape)
                anchor_grid = (anchor * stride).view((1, self.na, 1, 1, 2)).expand(shape)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), -1)
                xy = (xy * 2 - 0.5 + grid) * stride  # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)
