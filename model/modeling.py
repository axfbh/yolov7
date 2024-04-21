import torch.nn as nn
from ops.model.backbone.elandarknet53 import ElanDarkNet53
from ops.model.backbone.utils import _elandarknet_extractor
from ops.model.neck.spp import SPPCSPC
from ops.model.neck.panet import PaNet
from ops.model.head.yolo_head import YoloHead
from ops.model.neck.yolo_neck import YVBlock
from ops.model.misc.rep_conv import RepConv2d


class YoloV7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = _elandarknet_extractor(ElanDarkNet53(), 5)

        self.sppcspc = SPPCSPC(1024, 512, activation_layer=nn.SiLU)

        self.neck = PaNet([512, 256, 128], YVBlock, activation_layer=nn.SiLU)

        self.cv1 = RepConv2d(512, 1024)
        self.cv2 = RepConv2d(256, 512)
        self.cv3 = RepConv2d(128, 256)

        self.head = YoloHead([256, 512, 1024], num_classes)

    def forward(self, x):
        x = self.backbone(x)

        p7 = self.sppcspc(x['2'])

        sample = [p7, x['1'], x['0']]

        p5, p6, p7 = self.neck(sample)

        p5 = self.cv3(p5)
        p6 = self.cv2(p6)
        p7 = self.cv1(p7)

        head = self.head([p5, p6, p7])

        return head


def get_model(args):
    return YoloV7(3 * (5 + args.num_classes))
