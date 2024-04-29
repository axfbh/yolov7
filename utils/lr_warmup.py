import math
from typing import List
import numpy as np


class WarmupLR:
    def __init__(self,
                 optimizer,
                 scheduler,
                 last_iter: int,
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
        self.last_iter = last_iter

    def step(self):
        self.last_iter += 1
        it = self.last_iter

        if it <= self.warmup_iter:
            xi = [0, self.warmup_iter]  # x interp

            for j, x in enumerate(self.optimizer.param_groups):
                x["lr"] = np.interp(it,
                                    xi,
                                    [self.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"]])
                if "momentum" in x:
                    x["momentum"] = np.interp(it,
                                              xi,
                                              [self.warmup_momentum, self.momentum])
