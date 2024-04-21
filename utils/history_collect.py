import os

import torch
import datetime

import matplotlib
from scipy.signal import savgol_filter
from typing import List, Tuple, Union

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def save_model(model, optimizer, metric):
    loss = metric['loss'].avg
    epoch = metric['epoch']
    path = f'./weight/model_epoch_{epoch}_loss_{loss}_.pth'
    save_dict = {
        'optimizer_name': optimizer.__class__.__name__,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'last_epoch': epoch
    }
    torch.save(save_dict, path)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_arr(self, num):
        self.val_arr = [0. for _ in range(num)]
        self.avg_arr = [0. for _ in range(num)]
        self.sum_arr = [0. for _ in range(num)]
        self.count_arr = [0. for _ in range(num)]

    def update(self, val: Union[float, int, List], n=1):

        # -------------- 数组统计 --------------
        if isinstance(val, List):

            if not hasattr(self, 'val_arr'):
                num = len(val)
                self.reset_arr(num)

            for i in range(len(val)):
                if val[i] is not None:
                    self.val_arr[i] = val[i]
                    self.sum_arr[i] = val[i] * n
                    self.count_arr[i] = n
                    self.avg_arr[i] = self.sum_arr[i] / self.count_arr[i]

        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def __repr__(self):
        if hasattr(self, 'val_arr'):
            return str(self.avg_arr)
        return str(self.avg)


class HistoryLoss:
    def __init__(self, log_dir: str, log_num: int, modes: List[str]):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        # ------------- 根据 时间创建文件夹 ---------------
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.mode_dir = []
        for mode in modes:
            self.mode_dir.append(os.path.join(log_dir, mode))
            if not os.path.exists(self.mode_dir[-1]):
                os.makedirs(self.mode_dir[-1])

        self.loss_arrs = [[] for _ in modes]
        self.log_num = log_num

    def append(self, *args):
        for loss_arr, l in zip(self.loss_arrs, args):
            loss_arr.append(l)

    def loss_plot(self, start, log_num=None):
        for loss, path in zip(self.loss_arrs, self.mode_dir):
            log_num = self.log_num if log_num is None else log_num

            if len(loss) > log_num:
                iters = range(start, len(loss) + start)
                plt.figure()
                plt.plot(iters, loss, 'red', linewidth=2, label='train loss')
                try:
                    num = 5 if len(loss) < 25 else 10
                    plt.plot(iters, savgol_filter(loss, num, 3), 'green', linestyle='--', linewidth=2,
                             label='smooth train loss')
                except Exception:
                    pass
                else:
                    plt.grid(True)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend(loc="upper right")

                    plt.savefig(os.path.join(path, "epoch_loss.png"))

                    plt.cla()
                    plt.close("all")

    def __len__(self):
        return len(self.loss_arrs)