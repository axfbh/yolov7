import torch.nn as nn
import torch


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def warmup(self):
        s = 256  # 2x min stride
        forward = lambda x: self.forward(x)
        self.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  # forward

    def activations_hook(self, grad):
        # 获取梯度的钩子
        self.gradients = grad
