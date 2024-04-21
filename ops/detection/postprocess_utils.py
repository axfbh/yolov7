import torch.nn as nn
import torch
from abc import abstractmethod
from ops.detection.utils import make_grid
from ops.detection.nms import non_max_suppression
from typing import Sequence
from ops.iou import box_convert
from math import ceil


class PostProcess(nn.Module):
    def __init__(self, device, conf_thres=0.2, iou_thres=0.6):
        super(PostProcess, self).__init__()
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.device = device

    @abstractmethod
    def handle_preds(self, preds: torch.Tensor, anchors: torch.Tensor, image_size: Sequence):
        raise NotImplemented

    def forward(self, preds, anchors, image_size):
        inf_out = self.handle_preds(preds, anchors, image_size)
        return non_max_suppression(inf_out, self.conf_thres, self.iou_thres)


class YoloPostProcess(PostProcess):
    def __init__(self, device, conf_thres=0.2, iou_thres=0.6):
        super(YoloPostProcess, self).__init__(device, conf_thres, iou_thres)

    def handle_preds(self, preds, anchors, image_size):
        inf_out = []

        grids = [torch.as_tensor(pi.shape[-2:], device=self.device) for pi in preds]

        g = 1

        for i, pi in enumerate(preds):
            bs = pi.size(0)

            stride = torch.tensor(image_size, device=self.device) / grids[i]

            # 锚框下采样到特征图尺寸
            anchor = anchors[i] / stride[[1, 0]]

            # 更改 特征图 形式
            pi = pi.reshape(bs, len(anchor), -1, grids[i][0], grids[i][1])
            pi = pi.permute(0, 1, 3, 4, 2).contiguous()

            nc = pi.size(-1)

            # 创建网格
            grid_xy = make_grid(grids[i][0], grids[i][1], 1, 1, self.device).view(1, 1, grids[i][0], grids[i][1], 2)

            io = pi.clone()

            # 映射到网格
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) * (1 + ceil(g / 0.5)) - g + grid_xy
            io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * anchor.view(1, len(anchor), 1, 1, 2)
            io[..., 4:] = torch.sigmoid(io[..., 4:])

            # 将归一化的预测框,恢复到输入图像尺寸
            io[..., :4] *= stride[[1, 0, 1, 0]]

            # Compute conf : conf = obj_conf * cls_conf
            io[..., 5:] *= io[..., 4:5]

            cls_scores, _ = torch.max(io[..., 5:], dim=-1)

            io[..., 4] = cls_scores

            io = io.view(bs, -1, nc)

            inf_out.append(io)
        return torch.cat(inf_out, 1)


class FcosPostProcess(PostProcess):
    def __init__(self, device, conf_thres=0.2, iou_thres=0.6):
        super(FcosPostProcess, self).__init__(device, conf_thres, iou_thres)

    def handle_preds(self, preds, anchors, image_size):
        inf_out = []

        for i, pred in enumerate(preds):
            # 更改 特征图 形式
            pi = torch.cat(pred, 1)

            pi = pi.permute(0, 2, 3, 1).contiguous()

            bs, ny, nx, nc = pi.size()

            anchor = anchors[i]

            anchor = anchor.reshape([1, ny, nx, 4])

            pxy = 0.5 * (anchor[..., :2] + anchor[..., 2:])

            pwh = anchor[..., 2:] - anchor[..., :2]

            io = pi.clone()

            # 映射到网格
            io[..., 0:2] = pxy - (io[..., 0:2] * pwh)
            io[..., 2:4] = pxy + (io[..., 2:4] * pwh)
            io[..., 0:4] = box_convert(io[..., 0:4], in_fmt='xyxy', out_fmt='cxcywh')

            io[..., 4:] = torch.sigmoid(io[..., 4:])

            cls_scores, _ = torch.max(io[..., 5:], dim=-1)

            io[..., 4] = (io[..., 4] * cls_scores).sqrt()

            io = io.view(bs, -1, nc)

            inf_out.append(io)
        return torch.cat(inf_out, 1)
