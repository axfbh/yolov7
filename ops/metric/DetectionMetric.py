import torch
from ops.iou import bbox_iou, cxcwh2xy
import numpy as np


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def _ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i],
                              recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


class DetectionMetric:
    def __init__(self, device):
        self.stats = []

        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        self.iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.device = device

    def __call__(self, *args, **kwargs):
        """

        :param batchs: 每个 batch 的 non_max_suppression 的结果
        :param targets: 每个 batch 的 标签
        :param heights: 每个 batch 的图片 长
        :param widths: 每个 batch 的图片 宽
        :return:
        """
        output = args[0]
        target = args[1]
        height = args[2]
        width = args[3]

        whwh = torch.tensor([width, height, width, height], device=self.device)

        for i_cls, pred in enumerate(output):
            # ----------- 选取对应类别的 标签 ----------
            labels = target[target[:, 0] == i_cls, 1:]

            nl = len(labels)

            # ----------- 取出 类别 标签 ----------
            tcls = labels[:, 0].tolist() if nl else []

            # ----------- 统计 没有预测框，但是 有标签 的情况-----------
            if pred is None:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool),
                                       torch.Tensor(),
                                       torch.Tensor(),
                                       tcls))
                continue

            # ---------- 裁剪 预测框 超出 图像大小的部分 -----------
            clip_coords(pred, (height, width))

            # ---------- 统计 预测正确的情况 -----------
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)

            if nl:
                # --------- 已经被选取的目标 索引 -----------
                detected = []  # target indices

                # ------------  取出 类别 标签，保持 tensor 类型------------
                tcls_tensor = labels[:, 0]

                # ----------- 将中心值 转换回 xyxy值，并且映射到 image size 大小 -----------
                tbox = cxcwh2xy(labels[:, 1:5]) * whwh

                for cls in torch.unique(tcls_tensor):
                    # ----------- 取出 标签中 对应类别的索引 -----------
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices

                    # ----------- 取出 预测中 对应类别的索引 -----------
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    if pi.shape[0]:
                        # ----------- 同一个 类别 标签的情况下，哪个预测框和哪个标签框的iou最大，做匹配操作，一对一匹配 ------------
                        ious, i = bbox_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # 预测和标签框的 iou 必须大于 0.5
                        for j in (ious > self.iouv[0]).nonzero():
                            d = ti[i[j]]
                            # ----------- 一个真实框 只能匹配 一个预测框，一对一匹配，记录哪个真实框已经匹配了 -----------
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    def ap_per_class(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        mp, mr, map, mf1 = 0., 0., 0., 0.
        if len(stats):
            p, r, ap, f1, ap_class = _ap_per_class(*stats)
            if self.niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        return mp, mr, map, mf1
