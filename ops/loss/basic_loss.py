import torch
import torch.nn as nn
from ops.utils.torch_utils import de_parallel


class BasicLoss(nn.Module):
    def __init__(self, model):
        super(BasicLoss, self).__init__()
        self.device = next(model.parameters()).device
        self.hyp = de_parallel(model).hyp
