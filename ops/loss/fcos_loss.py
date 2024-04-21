import numpy as np
import torch
import torch.nn as nn
from ops.iou import iou_loss
# from ops.loss.focal_loss import py_sigmoid_focal_loss
from torchvision.ops.focal_loss import sigmoid_focal_loss
from ops.detection.anchor_utils import AnchorGenerator
from collections import Counter

torch.set_printoptions(precision=4, sci_mode=False)


class Fcosloss(nn.Module):
    def __init__(self, sigma, args):
        super(Fcosloss, self).__init__()
        self.num_classes = args.num_classes
        self.device = args.device.train

        self.sigma = sigma
        self.anchor_generator = AnchorGenerator([8, 16, 32, 64, 128], self.device)

        self.lower_bounds = self.anchor_generator.sizes * 4
        self.lower_bounds[0] = 0
        self.upper_bounds = self.anchor_generator.sizes * 8
        self.upper_bounds[-1] = torch.inf

    def build_targets(self, targets, grids, image_size):
        tcls, tbox, tcnt, indices, anchs = [], [], [], [], []

        anchors = self.anchor_generator(image_size, grids)

        strides = self.anchor_generator.strides

        for i in range(len(grids)):

            stride = strides[i].flip(0)

            anchor = anchors[i]

            anchor_centers = (anchor[:, :2] + anchor[:, 2:]) / 2  # N

            anchor_sizes = anchor[0, 2] - anchor[0, 0]

            radius = anchor_sizes * self.sigma

            limit_range = [self.lower_bounds[i], self.upper_bounds[i]]

            x, y = anchor_centers.chunk(2, 1)

            identity = torch.zeros_like(x)

            t = torch.Tensor(size=(0, 8)).to(self.device)

            # ------------- 每个 image 单独制作gt -------------
            for tb in targets:
                nb, cls, x0, y0, x1, y1 = tb.unbind(1)

                # ------------ 所有网格 与 目标的左上角和右下角计算 偏移量 ------------
                # l_off[N,M]: x[N,1] ,x0[M]
                tb = torch.stack([nb - identity,
                                  cls - identity,
                                  x0 - identity,
                                  y0 - identity,
                                  x - x0,
                                  y - y0,
                                  x1 - x,
                                  y1 - y], dim=-1)

                ltrb_off = tb[..., 4:]
                off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
                off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]

                # ------------- 所有网格 与 目标中心点计算 距离 -------------
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                c_ltrb_off = torch.stack([x - cx, y - cy, cx - x, cy - y], dim=-1)
                c_off_max = torch.max(c_ltrb_off, dim=-1)[0]

                # limit_range：满足论文 m 要求
                # radiu: 选择，中心点半径内的点帮助预测
                j = (off_max > limit_range[0]) & \
                    (off_max < limit_range[1]) & \
                    (off_min > 0) & (c_off_max < radius)

                # ------------- 目标面积 -------------
                areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])

                # ------------- 不符合要求的面积设置最小 -------------
                areas = j * (1e8 - areas)

                areas_min_ind = torch.max(areas, dim=-1)[1]

                tb = tb[
                    torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)
                ]

                j = j.sum(1) >= 1

                tb = tb[j]

                t = torch.cat([t, tb], 0)

            b, c = t[:, :2].long().t()

            off = t[:, 2:4]

            gxy = t[:, 4:6] + off

            gltrb = t[:, 4:8]

            gi, gj = (gxy / stride).long().t()

            l_r_min = torch.min(gltrb[:, 0], gltrb[:, 2])
            l_r_max = torch.max(gltrb[:, 0], gltrb[:, 2])
            t_b_min = torch.min(gltrb[:, 1], gltrb[:, 3])
            t_b_max = torch.max(gltrb[:, 1], gltrb[:, 3])
            cnt_targets = ((l_r_min * t_b_min) / (l_r_max * t_b_max + 1e-10)).sqrt()

            gbox = torch.cat([gxy[:, :2] - gltrb[:, :2], gxy[:, :2] + gltrb[:, 2:]], -1)

            indices.append([b, gj, gi])

            tbox.append(gbox)

            tcls.append(c)

            tcnt.append(cnt_targets)

            anchs.append([gxy, anchor_sizes])

        return tcls, tbox, tcnt, anchs, indices

    def forward(self, preds, targets, image_size):
        grids = [torch.as_tensor(pi[0].shape[-2:], device=self.device) for pi in preds]

        BCE = nn.BCEWithLogitsLoss(reduction='sum')

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lcnt = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, tcnt, anchs, indices = self.build_targets(targets, grids, image_size)

        n = 0

        for i, pred in enumerate(preds):
            # 1：reg_head
            # 2：cnt_head
            # 3：cls_head
            pi = torch.cat(pred, 1)
            pi = pi.permute([0, 2, 3, 1]).contiguous()

            b, gj, gi = indices[i]

            nb = len(b)

            tobj = torch.zeros_like(pi[..., 5:])

            n += nb

            if nb:
                ps = pi[b, gj, gi]
                tobj[b, gj, gi, tcls[i]] = 1.0

                pxy, anch_wh = anchs[i]

                pwh = ps[:, :4] * anch_wh

                pbox = torch.cat([pxy - pwh[:, :2], pxy + pwh[:, 2:]], dim=-1)

                giou = iou_loss(pbox, tbox[i], GIoU=True)

                lbox += (1.0 - giou).sum()

                lcnt += BCE(ps[:, 4], tcnt[i])

            lcls += sigmoid_focal_loss(pi[..., 5:], tobj, reduction='sum')

        lbox /= n
        lcnt /= n
        lcls /= n

        loss = lbox + lcnt + lcls

        return loss, lbox.detach(), lcnt.detach(), lcls.detach()