from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer
import math
from bisect import bisect_right
from typing import List

import warnings

def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0
    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def get_multistep_lr(base_lr, warmup_factor, gamma, milestones, current_epoch):
    return base_lr * warmup_factor * gamma ** bisect_right(milestones, current_epoch)


def get_cosine_lr(base_lr, warmup_factor, current_epoch, end_epoch):
    return base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * current_epoch / end_epoch))


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_epoch=-1,
                 warmup_method: str = 'linear',
                 warmup_factor: float = 0.001,
                 warmup_iters: int = 10):

        if warmup_factor > 1.:
            raise ValueError('warmup_factor should be less than or equal to 1.')

        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )

        self.gamma = gamma
        self.milestones = milestones
        self.last_epoch = last_epoch
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters

        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor)

        return [get_multistep_lr(base_lr,
                                 warmup_factor,
                                 self.gamma,
                                 self.milestones,
                                 self.last_epoch) for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 end_epoch,
                 last_epoch=-1,
                 warmup_method: str = "linear",
                 warmup_factor: float = 0.001,
                 warmup_iters: int = 10):
        self.end_epoch = end_epoch
        self._last_lr = None
        self.last_epoch = last_epoch
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters

        if self.warmup_factor > 1.:
            raise ValueError('warmup_factor should be less than or equal to 1.')

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor)

        return [get_cosine_lr(base_lr,
                              warmup_factor,
                              self.last_epoch,
                              self.end_epoch) for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class PolynomialLRWarmup(_LRScheduler):
    def __init__(self, optimizer,
                 warmup_iters,
                 total_iters=5,
                 power=1.0,
                 last_epoch=-1,
                 verbose=False):
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch <= self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            l = self.last_epoch
            w = self.warmup_iters
            t = self.total_iters
            decay_factor = ((1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):

        if self.last_epoch <= self.warmup_iters:
            return [
                base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return [
                (
                        base_lr * (1.0 - (min(self.total_iters, self.last_epoch) - self.warmup_iters) / (
                        self.total_iters - self.warmup_iters)) ** self.power
                )
                for base_lr in self.base_lrs
            ]
