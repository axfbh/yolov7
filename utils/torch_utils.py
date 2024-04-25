import torch
import os
from typing import List
import torch.nn as nn
from torch.nn.parameter import is_lazy
from utils.logging import LOGGER, colorstr


def _check_file_exits(path):
    return not isinstance(path, bool) and os.path.isfile(path)


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


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
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


def smart_resume(model, optimizer, save_path=None):
    last_epoch = -1
    # ------------ resume model ------------
    if _check_file_exits(save_path):
        save_dict = torch.load(save_path, map_location='cpu')

        last_epoch = save_dict.get('last_epoch', None)

        # ---------- 加载模型权重 ----------
        model_param = save_dict['model']
        _load_from(model, model_param)

        # ---------- epoch 识别 ----------
        resume_info = last_epoch if last_epoch is not None else save_path.strip('\n\t').split('_')[2]
        try:
            last_epoch = int(resume_info)
        except TypeError:
            last_epoch = -1
            LOGGER.warning(
                f"{colorstr('Warning')} cannot loaded the previous last_epoch , but it doesnt affect the model working."
            )
            LOGGER.info(
                f"{colorstr('model loaded:')} pate {save_path} -> {model._get_name()}"
            )
        else:
            LOGGER.info(
                f"{colorstr('model loaded:')} path: {save_path} last_epoch: {last_epoch} -> {model._get_name()}"
            )

    # ------------ resume optimizer ------------
    if _check_file_exits(save_path):
        save_dict = torch.load(save_path, map_location='cpu')

        optim_param = save_dict.get('optimizer', None)
        optim_name = save_dict.get('optimizer_name', None)

        if optim_name == optimizer.__class__.__name__:
            optimizer.load_state_dict(optim_param)
            for param in optimizer.param_groups:
                LOGGER.info(
                    f"{colorstr('optimizer loaded:')} {type(optimizer).__name__}(lr={param['lr']}) with parameter groups"
                    f"{len(param)} weight(decay={param['weight_decay']})"
                )
            return optimizer
        else:
            LOGGER.warning(
                f"{colorstr('Warning')} cannot loaded the optimizer parameter into corresponding optimizer , but it doesnt affect the model working."
            )
    return last_epoch
