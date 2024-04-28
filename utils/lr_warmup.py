import math
from typing import List
import numpy as np


def warmup_factor_at_iter(optimizer,
                          scheduler,
                          it: int,
                          epoch: int,
                          momentum: float,
                          warmup_iter: int,
                          warmup_bias_lr: float,
                          warmup_momentum: float):
    """
    Args:
        optimizer : 学习率优化器
        scheduler : 学习率调整器
        it (int): number integrated batches (since train start)
        epoch (int): the number of epoch.
        momentum (float): optimizer momentum
        warmup_iter (int): the number of warmup iterations.
        warmup_bias_lr (float): warmup bias learning
        warmup_momentum (float): warmup momentum.

    """
    # bias lr falls from 0.1 to lr0
    # all other lrs rise from 0.0 to lr0
    if it > warmup_iter:
        return

    last_lr = scheduler.get_last_lr()
    for j, x in enumerate(optimizer.param_groups):
        x["lr"] = np.interp(it,
                            [0, warmup_iter],
                            [warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * last_lr[j]])
        if "momentum" in x:
            x["momentum"] = np.interp(it, [0, warmup_iter], [warmup_momentum, momentum])


class WarmupLR:
    def __init__(self,
                 optimizer,
                 scheduler,
                 epoch: int,
                 momentum: float,
                 warmup_iter: int,
                 warmup_bias_lr: float,
                 warmup_momentum: float):
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_iter = warmup_iter
        self.momentum = momentum
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.epoch = epoch

    def step(self, it):

        if it > self.warmup_iter:
            return

        last_lr = self.scheduler.get_last_lr()
        for j, x in enumerate(self.optimizer.param_groups):
            x["lr"] = np.interp(it,
                                [0, self.warmup_iter],
                                [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * last_lr[j]])
            if "momentum" in x:
                x["momentum"] = np.interp(it, [0, self.warmup_iter], [self.warmup_momentum, self.momentum])
