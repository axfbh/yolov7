import torch.nn as nn
import torch
from ops.model.detection.yolov7 import YoloV7
from ops.model.detection.yolov4 import YoloV4
from ops.model.detection.yolov3 import YoloV3


def get_model(cfg):
    return YoloV4(anchors=cfg.anchors, num_classes=cfg.nc, phi='l')
