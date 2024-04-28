import os
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import is_lazy
from torchvision.ops.boxes import box_convert

from utils.logging import LOGGER, colorstr


@torch.no_grad()
def _load_from(model, weight):
    model_state_dict = model.state_dict()

    for k in list(weight.keys()):
        if k in model_state_dict:
            if is_lazy(model_state_dict[k]):
                continue
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(weight[k].shape)
            if shape_model != shape_checkpoint:
                weight.pop(k)
        else:
            weight.pop(k)
            print(k)

    model.load_state_dict(weight)


def smart_optimizer(model, name: str = "Adam", lr=0.001, momentum=0.9, decay=1e-5):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer


def smart_scheduler(optimizer, name: str = "Cosine", last_epoch=1, **kwargs):
    if name == "Cosine":
        # T_max:end_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               last_epoch=last_epoch,
                                                               **kwargs)
    elif name == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         last_epoch=last_epoch,
                                                         **kwargs)
    elif name == "Polynomial":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                          last_epoch=last_epoch,
                                                          **kwargs)
    elif name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        last_epoch=last_epoch,
                                                        **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    args = {k: v for k, v in kwargs.items()}
    LOGGER.info(
        f"{colorstr('scheduler:')} " + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"
    )
    return scheduler


def smart_resume(model, optimizer, ema=None, epochs=300, resume=False, save_path: Path = None):
    last_epoch = -1
    best_fitness = None
    if not resume or save_path is None or not save_path.is_file():
        LOGGER.warning(
            f"{colorstr('Warning:')} Noting to resume"
        )
        start_epoch = last_epoch + 1
        return last_epoch, best_fitness, start_epoch, epochs
    save_dict = torch.load(save_path, map_location='cpu')

    # ------------ resume model ------------
    model_param = save_dict['model'].state_dict()
    _load_from(model, model_param)

    last_epoch = save_dict.get('last_epoch', last_epoch)
    start_epoch = last_epoch + 1

    best_fitness = save_dict.get('best_fitness', best_fitness)

    LOGGER.info(
        f"{colorstr('model loaded:')} Resuming training from {save_path} from epoch {start_epoch} to {epochs} total epochs"
    )

    eam_param = save_dict.get('ema', None)

    if ema and eam_param is not None:
        ema.ema.load_state_dict(eam_param.state_dict())  # EMA
        ema.updates = save_dict["updates"]

    # ------------ resume optimizer ------------

    optim_param = save_dict.get('optimizer', None)
    optim_name = save_dict.get('optimizer_name', None)

    if optim_name == optimizer.__class__.__name__ and optim_param is not None:
        optimizer.load_state_dict(optim_param)
        for param in optimizer.param_groups:
            LOGGER.info(
                f"{colorstr('optimizer loaded:')} {type(optimizer).__name__}(lr={param['lr']}) with parameter groups"
                f"{len(param)} weight(decay={param['weight_decay']})"
            )
    else:
        LOGGER.warning(
            f"{colorstr('Warning')} Cannot loaded the optimizer parameter, but it doesnt affect the model working."
        )

    return last_epoch, best_fitness, start_epoch, epochs


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()


def output_to_target(output, max_det=300):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box_convert(box, 'xyxy', 'cxcywh'), conf), 1))
    return torch.cat(targets, 0).numpy()
