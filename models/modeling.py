import torch.nn as nn
import torch
from ops.models.detection.yolov7 import YoloV7
from ops.models.detection.yolov5 import YoloV5
from ops.models.detection.yolov4 import YoloV4
from ops.models.detection.fcos import FCOS


def get_model(cfg):
    return YoloV7(anchors=cfg.anchors, num_classes=cfg.nc, phi='l')
    # return YoloV5(anchors=cfg.anchors,
    #               num_classes=cfg.nc,
    #               depth_multiple=cfg.depth_multiple,
    #               width_multiple=cfg.width_multiple)
    # return FCOS(num_classes=cfg.nc, anchors=cfg.anchors, aspect_ratios=cfg.aspect_ratios)
