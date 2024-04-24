import torch
import os
from typing import List
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


class WeightInject:
    def __init__(self,
                 model,
                 optimizer,
                 params,
                 device):
        """

        :param model:
        :param optimizer:
        :param params:
        """
        self.device = device
        self.optimizer = optimizer
        self.model = model
        self.resume_path = params['resume']
        self.print_info = ''
        self.last_epoch = -1
        self.lr = [param['lr'] for param in optimizer.param_groups]

    def load_state_dict(self):
        self.__weight_load()

    def __weight_load(self):
        if _check_file_exits(self.resume_path):
            save_dict = torch.load(self.resume_path, map_location=self.device)
            # ---------- 加载模型权重 ----------
            model_param = save_dict['model']
            _load_from(self.model, model_param)

            # ---------- 加载优化器权重 ----------
            optim_param = save_dict.get('optimizer', None)
            optim_name = save_dict.get('optimizer_name', None)
            last_epoch = save_dict.get('last_epoch', None)

            if optim_param is None or optim_name is None:
                LOGGER.warning(
                    f"{colorstr('Warning')} cannot loaded the previous optimizer parameter , but it doesnt affect the model working."
                )

            if optim_name == self.optimizer.__class__.__name__:
                self.optimizer.load_state_dict(optim_param)
                for param in self.optimizer.param_groups:
                    LOGGER.info(
                        f"{colorstr('optimizer loaded:')}  lr: {param['lr']} lr: {param['weight_decay']}"
                    )
            else:
                LOGGER.warning(
                    f"{colorstr('Warning')} cannot loaded the optimizer parameter into corresponding optimizer , but it doesnt affect the model working."
                )

            # ---------- epoch 识别 ----------
            resume_info = last_epoch if last_epoch is not None else self.resume_path.strip('\n\t').split('_')[2]
            try:
                self.last_epoch = int(resume_info)
            except TypeError:
                LOGGER.warning(
                    f"{colorstr('Warning')} cannot loaded the previous last_epoch , but it doesnt affect the model working."
                )
                LOGGER.info(
                    f"{colorstr('model loaded:')} pate {self.resume_path} -> {self.model._get_name()}"
                )
            else:
                LOGGER.info(
                    f"{colorstr('model loaded:')} path: {self.resume_path} last_epoch: {self.last_epoch} -> {self.model._get_name()}"
                )
        else:
            self.print_info = 'nothing to loading'
