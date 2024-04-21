import torch
import torch.nn as nn


class ConvolutionLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None):
        p = (k - 1) // 2 if p is None else p
        super(ConvolutionLayer, self).__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )


class Elan(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=0.5):
        super(Elan, self).__init__()

        mid_ch = int(in_ch * expand_ratio)

        self.cv1 = ConvolutionLayer(in_ch, mid_ch, 1, 1)
        self.cv2 = ConvolutionLayer(in_ch, mid_ch, 1, 1)

        self.cv3 = nn.Sequential(ConvolutionLayer(mid_ch, mid_ch, 3, 1),
                                 ConvolutionLayer(mid_ch, mid_ch, 3, 1))

        self.cv4 = nn.Sequential(ConvolutionLayer(mid_ch, mid_ch, 3, 1),
                                 ConvolutionLayer(mid_ch, mid_ch, 3, 1))

        self.cv5 = ConvolutionLayer(mid_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)

        x2 = self.cv2(x)

        x3 = self.cv3(x1)

        x4 = self.cv4(x3)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return self.cv5(x)


class ElanH(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=0.5):
        super(ElanH, self).__init__()

        mid_ch = int(in_ch * expand_ratio)
        mid2_ch = int(mid_ch * expand_ratio)

        self.cv1 = ConvolutionLayer(in_ch, mid_ch, 1, 1)
        self.cv2 = ConvolutionLayer(in_ch, mid_ch, 1, 1)

        self.cv3 = ConvolutionLayer(mid_ch, mid2_ch, 3, 1)
        self.cv4 = ConvolutionLayer(mid2_ch, mid2_ch, 3, 1)

        self.cv5 = ConvolutionLayer(mid2_ch, mid2_ch, 3, 1)
        self.cv6 = ConvolutionLayer(mid2_ch, mid2_ch, 3, 1)

        self.cv7 = ConvolutionLayer(mid_ch * 2 + mid2_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)

        x2 = self.cv2(x)

        x3 = self.cv3(x1)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        return self.cv7(x)


if __name__ == '__main__':
    image = torch.randn((1, 128, 13, 13))

    elan = ElanH(128, 256)

    print(elan(image))
