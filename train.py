import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import torch

from models.modeling import get_model
from data_loader import get_loader

from ops.loss.yolo_loss import YoloLossV7
from utils.epoch_utils import train_epoch, val_epoch
from utils.history_collect import History
from utils.torch_utils import smart_optimizer, smart_resume, smart_scheduler
from utils.logging import print_args
from utils.torch_utils import de_parallel


# hyp: hyper parameter
# opt: options
def train(train_loader, val_loader, hyp, opt):
    cfg = OmegaConf.load(Path(opt.cfg))

    nb = opt.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / nb), 1)

    device = opt.device
    model = get_model(cfg)
    model.to(device)

    m = de_parallel(model).head  # number of detection layers (to scale hyps)
    nl = m.nl
    nc = m.num_classes
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (max(opt.image_size[0], opt.image_size[1]) / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.hyp = hyp

    # -------- 梯度优化器 --------
    optimizer = smart_optimizer(model,
                                opt.optimizer,
                                hyp.lr,
                                hyp.momentum,
                                hyp.weight_decay)

    # -------- 梯度优化器 --------
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -------- 模型权重加载器 --------
    last_epoch = smart_resume(model, optimizer, Path(opt.resume))

    start_epoch = last_epoch + 1
    end_epoch = opt.epochs

    # -------- 学习率优化器 and 学习率预热器 --------
    scheduler = smart_scheduler(optimizer,
                                opt.scheduler,
                                last_epoch,
                                end_epoch=end_epoch,
                                warmup_method=hyp.warmup_method,
                                warmup_factor=hyp.warmup_factor,
                                warmup_iters=hyp.warmup_iters)

    # -------- 每个 epoch 的 loss 记录工具 --------
    history = History(project_dir=Path(opt.project),
                      name=opt.name,
                      mode='train',
                      save_period=opt.save_period,
                      yaml_args={'hyp': hyp, 'opt': vars(opt)})

    for epoch in range(start_epoch, end_epoch):
        train_metric = train_epoch(model=model,
                                   loader=train_loader,
                                   device=device,
                                   epoch=epoch,
                                   optimizer=optimizer,
                                   criterion=YoloLossV7(model),
                                   scaler=scaler,
                                   accumulate=accumulate)

        val_metric = val_epoch(model=model,
                               loader=val_loader,
                               device=device,
                               epoch=epoch,
                               criterion=YoloLossV7(model))

        scheduler.step()

        history.save(model, optimizer, epoch, val_metric['map50'])


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--cfg", type=str, default="models/yolov7l.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./config/hyp-yolo-v7-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", default='./logs/train/exp1/weights/last.pt', help="resume most recent training")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--scheduler", type=str, choices=["Cosine", "MultiStep", "Polynomial"], default="Cosine",
                        help="scheduler")
    parser.add_argument("--workers", type=int, default=3, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./logs", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--save-period", type=int, default=3, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    return parser.parse_args()


def main():
    opt = parse_opt()
    print_args(vars(opt))

    hyp = OmegaConf.load(Path(opt.hyp))

    train_loader, val_loader = get_loader(hyp, opt)

    train(train_loader, val_loader, hyp, opt)


if __name__ == '__main__':
    print('---------------')
    main()
