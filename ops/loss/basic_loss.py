import torch
import torch.nn as nn


class BasicLoss(nn.Module):
    def __init__(self, model):
        super(BasicLoss, self).__init__()
        self.device = next(model.parameters()).device
        self.hyp = model.hyp
