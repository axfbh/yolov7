import torch
import torch.nn as nn
from abc import abstractmethod
from ops.iou import bbox_iou, iou_loss
from math import ceil

torch.set_printoptions(precision=4, sci_mode=False)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YoloLoss(nn.Module):
    anchors = None

    def __init__(self, args, thresh=0.6):
        super(YoloLoss, self).__init__()
        self.thresh = thresh
        self.cp, self.cn = smooth_BCE(eps=0.0)
        self.num_classes = args.num_classes
        self.device = args.train.device
        # yolo 大网格：小锚框，小网格：大锚框

        # yolo v3 是小grid先出，大grid后出
        self.anchors = torch.tensor(args.anchors).view(3, -1, 2).to(self.device)

    @abstractmethod
    def build_targets(self, targets, grids, image_size):
        raise NotImplemented


class YoloLossV3(YoloLoss):
    def build_targets(self, targets, grids, image_size):

        tcls, txy, twh, indices = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(len(grids)):
            # ----------- 提取 不同尺度的 网格大小 -----------
            ng = grids[i]

            stride = image_size / ng

            # ------------- 将 anchor 映射到 grid 的大小 -------------
            anchor = self.anchors[i] / stride[[1, 0]]

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 7)).to(self.device)

            for tb in targets * gain:
                # ----------- 计算 锚框 与 长宽 的 iou -----------
                gwh = tb[:, 4:6]
                iou = bbox_iou(anchor, gwh, in_fmt='wh')
                iou, a = iou.max(0)

                # ------------ 删除小于阈值的框 -------------
                j = iou.view(-1) > self.thresh
                tb, a = tb[j], a[j]

                tb = torch.cat([tb, a[:, None]], -1)

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 4:6]

            a = t[:, 6].long()

            gi, gj = gxy.long().t()

            indices.append([b, a, gj, gi])

            txy.append(gxy % 1)

            twh.append(torch.log(gwh / anchor[a]))

            tcls.append(c)

        return tcls, txy, twh, indices

    def forward(self, preds, targets, image_size):

        grids = [torch.as_tensor(pi.shape[-2:], device=self.device) for pi in preds]

        MSE = nn.MSELoss()
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lxy = torch.zeros(1, dtype=torch.float32, device=self.device)
        lwh = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, txy, twh, indices = self.build_targets(targets, grids, image_size)

        for i, pi in enumerate(preds):
            pi = pi.reshape(pi.size(0), len(self.anchors[0]), -1, grids[i][0], grids[i][1])
            pi = pi.permute(0, 1, 3, 4, 2).contiguous()

            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]
                tobj[b, a, gj, gi] = 1

                ptxy = torch.sigmoid(
                    ps[:, 0:2]
                )

                ptwh = ps[:, 2:4]

                # ------------ 计算 偏移量 差值 ------------
                lxy += MSE(ptxy, txy[i])
                lwh += MSE(ptwh, twh[i])

                # ------------ 计算 分类 loss ------------
                if self.num_classes > 1:
                    t = torch.zeros_like(ps[:, 5:])  # targets
                    t[range(nb), tcls[i] - 1] = 1
                    lcls += BCEcls(ps[:, 5:], t)

            # ------------ 计算 置信度 loss ------------
            lobj += BCEobj(pi[..., 4], tobj)

        lxy *= 11.3
        lwh *= 0.87
        lobj *= 69.2
        lcls *= 32

        loss = lxy + lwh + lobj + lcls

        return loss, lxy.detach(), lwh.detach(), lobj.detach(), lcls.detach()


class YoloLossV4(YoloLoss):

    def build_targets(self, targets, grids, image_size):
        """

        :param targets: 归一化的标签
        :param grids: 网格大小
        :return:
        """
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(len(grids)):
            # ----------- grid 大小 -----------
            ng = grids[i]

            # ----------- 图片与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 7)).to(self.device)

            for tb in targets * gain:
                # ----------- 计算 锚框 与 长宽 的 iou -----------
                gwh = tb[:, 4:6]
                iou = bbox_iou(anchor, gwh, in_fmt='wh')
                iou, a = iou.max(0)

                # ------------ 删除小于阈值的框 -------------
                j = iou.view(-1) > self.thresh
                tb, a = tb[j], a[j]

                tb = torch.cat([tb, a[:, None]], -1)

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 4:6]

            a = t[:, 6].long()

            gi, gj = gxy.long().t()

            indices.append([b, a, gj, gi])

            txy = gxy % 1

            tbox.append(torch.cat([txy, gwh], 1))

            anch.append(anchor[a])

            tcls.append(c)

        return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        grids = [torch.as_tensor(pi.shape[-2:], device=self.device) for pi in preds]

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(targets, grids, image_size)

        for i, pi in enumerate(preds):
            pi = pi.reshape(pi.size(0), len(self.anchors[0]), -1, grids[i][0], grids[i][1])
            pi = pi.permute([0, 1, 3, 4, 2]).contiguous()

            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]
                tobj[b, a, gj, gi] = 1

                pxy = torch.sigmoid(ps[:, 0:2])

                pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i]

                pbox = torch.cat([pxy, pwh], 1)

                giou = iou_loss(pbox, tbox[i], in_fmt='cxcywh', GIoU=True)

                lbox += (1.0 - giou).mean()

                if self.num_classes > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(nb), tcls[i] - 1] = self.cp
                    lcls += BCEcls(ps[:, 5:], t)

            lobj += BCEobj(pi[..., 4], tobj)

        lbox *= 3.54
        lobj *= 64.3
        lcls *= 37.4

        loss = lbox + lobj + lcls

        return loss, lbox.detach(), lobj.detach(), lcls.detach()


class YoloLossV7(YoloLoss):

    def __init__(self, args, g=1.0, thresh=4):
        super(YoloLossV7, self).__init__(args, thresh)
        self.g = g
        self.balance = [4.0, 1.0, 0.4]

    def build_targets(self, targets, grids, image_size):
        """

        :param image_size:
        :param targets: 归一化的标签
        :param grids: 网格大小
        :return:
        """
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(len(grids)):
            # ----------- grid 大小 -----------
            ng = grids[i]

            # ----------- 网格 ——----------
            x, y = torch.tensor([[0, 0],
                                 [1, 0],
                                 [0, 1],
                                 [-1, 0],
                                 [0, -1]], device=self.device, dtype=torch.float32).mul(self.g).chunk(2, 1)

            identity = torch.zeros_like(x)

            # ----------- 图片与 grid 的比值 -----------
            stride = image_size / ng

            # ----------- 锚框映射到 grid 大小 -----------
            anchor = self.anchors[i] / stride[[1, 0]]

            na = len(anchor)

            # ----------- 归一化的 坐标和长宽 -----------
            gain[2:] = (1 / stride)[[1, 0, 1, 0]]

            t = torch.Tensor(size=(0, 9)).to(self.device)

            for tb in targets * gain:
                nb, cls, cx, cy, gw, gh = tb.unbind(1)

                # ----------- 选择目标点 1 格距离内的网格用于辅助预测 -----------
                tb = torch.stack([nb - identity,
                                  cls - identity,
                                  cx - identity,
                                  cy - identity,
                                  cx - x,
                                  cy - y,
                                  gw - identity,
                                  gh - identity],
                                 -1)

                j = torch.bitwise_and(0 <= tb[..., 4:6], tb[..., 4:6] < ng[[1, 0]]).all(-1)
                tb = tb[j]

                # gxy = tb[..., 2:4]
                # gxi = ng[[1, 0]] - gxy
                # gxy % 1. < self.g，获得如下值
                # j：左格左上角
                # k：上格左上角
                # j, k = ((gxy % 1. < self.g) & (gxy > 1.)).unbind(-1)
                # gxi % 1. < self.g，获得如下值
                # l：右格左上角
                # m：下格左上角
                # l, m = ((gxi % 1. < self.g) & (gxi > 1.)).unbind(-1)
                # tb = tb[j]

                ai = torch.arange(na, device=self.device).view(na, 1).repeat(1, len(tb))

                tb = torch.cat((tb.repeat(na, 1, 1), ai[:, :, None]), 2)

                #  ------------ 选择最大的长宽比，删除小于阈值的框 -------------
                r = tb[..., 6:8] / anchor[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.thresh
                tb = tb[j]

                t = torch.cat([t, tb], 0)

            # ----------- 分别提取信息，生成 -----------
            b, c = t[:, :2].long().t()

            gxy = t[:, 2:4]

            gwh = t[:, 6:8]

            gij = t[:, 4:6].long()

            gi, gj = gij.t()

            a = t[:, 8].long()

            indices.append([b, a, gj, gi])

            tbox.append(torch.cat([gxy - gij, gwh], 1))

            anch.append(anchor[a])

            tcls.append(c)

        return tcls, tbox, indices, anch

    def forward(self, preds, targets, image_size):
        grids = [torch.as_tensor(pi.shape[-2:], device=self.device) for pi in preds]

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1, dtype=torch.float32, device=self.device))

        lcls = torch.zeros(1, dtype=torch.float32, device=self.device)
        lbox = torch.zeros(1, dtype=torch.float32, device=self.device)
        lobj = torch.zeros(1, dtype=torch.float32, device=self.device)

        tcls, tbox, indices, anchors = self.build_targets(targets, grids, image_size)

        for i, pi in enumerate(preds):
            pi = pi.reshape(pi.size(0), len(self.anchors[0]), -1, grids[i][0], grids[i][1])
            pi = pi.permute([0, 1, 3, 4, 2]).contiguous()

            b, a, gj, gi = indices[i]

            tobj = torch.zeros_like(pi[..., 0])

            nb = len(b)

            if nb:
                ps = pi[b, a, gj, gi]

                pxy = torch.sigmoid(ps[:, 0:2]) * (1 + ceil(self.g / 0.5)) - self.g

                pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]

                pbox = torch.cat([pxy, pwh], 1)

                iou = iou_loss(pbox, tbox[i], in_fmt='cxcywh', CIoU=True)

                lbox += (1 - iou).mean()

                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)

                if self.num_classes > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(nb), tcls[i] - 1] = self.cp
                    lcls += BCEcls(ps[:, 5:], t)

            lobj += BCEobj(pi[..., 4], tobj) * self.balance[i]

        lbox *= 3.54
        lobj *= 64.3
        lcls *= 37.4

        loss = lbox + lobj + lcls

        return loss, lbox.detach(), lobj.detach(), lcls.detach()