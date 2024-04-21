import torch
from functools import partial
import torch.nn as nn
from math import ceil

import torchvision.ops.boxes


def make_grid(h, w, device):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).view(1, 1, h, w, 2).to(device).float()


def handle_preds(preds, image_size, anchors, device):
    inf_out = []

    grids = [torch.as_tensor(pi[0].shape[-2:], device=device) for pi in preds]

    for i, pi in enumerate(preds):
        bs = pi.size(0)

        stride = torch.tensor(image_size) / grids[i]

        # 锚框下采样到特征图尺寸
        anchor = anchors[i] / stride

        # 更改 特征图 形式
        pi = pi.reshape(bs, len(anchor), -1, grids[i][0], grids[i][1])
        pi = pi.permute(0, 1, 3, 4, 2).contiguous()

        nc = pi.size(-1)

        # 创建网格
        grid_xy = make_grid(grids[i][0], grids[i][1], device)

        io = pi.clone()

        # 映射到网格
        io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + grid_xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor.view(1, len(anchor), 1, 1, 2)
        io[..., 4:] = torch.sigmoid(io[..., 4:])

        # 将归一化的预测框,恢复到输入图像尺寸
        io[..., :4] *= stride[[1, 0, 1, 0]]

        io = io.view(bs, -1, nc)

        inf_out.append(io)
    return torch.cat(inf_out, 1)
