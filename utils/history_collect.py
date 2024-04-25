import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import torch

from scipy.signal import savgol_filter

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')
import yaml


def yaml_save(file: Union[str, Path] = "data.yaml", data={}):
    # Single-line safe yaml saving
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return str(self.avg)


class History:
    def __init__(self, project_dir: Path, name: str, mode: str, save_period=-1, yaml_args: Dict = None):
        project_dir = project_dir.joinpath(mode)
        # ------------- 根据 时间创建文件夹 ---------------
        if not project_dir.exists():
            project_dir.mkdir()

        i = 0
        while True:
            exp_dir = project_dir.joinpath(name + str(i) if i else '')
            if not exp_dir.exists():
                exp_dir.mkdir()
                weight_dir = exp_dir.joinpath('weights')
                weight_dir.mkdir()
                break
            i += 1

        if yaml_args is not None:
            for k, v in yaml_args.items():
                yaml_save(exp_dir.joinpath(f"{k}.yaml"), v)

        self.exp_dir = exp_dir
        self.weight_dir = weight_dir
        self.best_fitness = None
        self.save_period = save_period

    def save(self, model, optimizer, epoch, fitness: float):
        if fitness >= self.best_fitness or self.best_fitness is None:
            self.best_fitness = fitness

        save_dict = {
            'last_epoch': epoch,
            "best_fitness": fitness,
            'optimizer_name': optimizer.__class__.__name__,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            "date": datetime.now().isoformat(),
        }

        # ---------- save last model --------------
        last_pt_path = self.weight_dir.joinpath('last.pt')
        torch.save(save_dict, last_pt_path)

        # ---------- save best model --------------
        if fitness == self.best_fitness:
            best_pt_path = self.weight_dir.joinpath('best.pt')
            torch.save(save_dict, best_pt_path)

        # ---------- save period model --------------
        if (epoch + 1) % self.save_period == 0:
            weights_pt_path = self.weight_dir.joinpath(f'weights{str(epoch)}.pt')
            torch.save(save_dict, weights_pt_path)

    # def append(self, *args):
    #     for loss_arr, l in zip(self.loss_arrs, args):
    #         loss_arr.append(l)
    #
    # def loss_plot(self, start, log_num=None):
    #     for loss, path in zip(self.loss_arrs, self.mode_dir):
    #         log_num = self.log_num if log_num is None else log_num
    #
    #         if len(loss) > log_num:
    #             iters = range(start, len(loss) + start)
    #             plt.figure()
    #             plt.plot(iters, loss, 'red', linewidth=2, label='train loss')
    #             try:
    #                 num = 5 if len(loss) < 25 else 10
    #                 plt.plot(iters, savgol_filter(loss, num, 3), 'green', linestyle='--', linewidth=2,
    #                          label='smooth train loss')
    #             except Exception:
    #                 pass
    #             else:
    #                 plt.grid(True)
    #                 plt.xlabel('Epoch')
    #                 plt.ylabel('Loss')
    #                 plt.legend(loc="upper right")
    #
    #                 plt.savefig(os.path.join(path, "epoch_loss.png"))
    #
    #                 plt.cla()
    #                 plt.close("all")
