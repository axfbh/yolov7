import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

import torch

import numpy as np

from models.modeling import get_model
from dataloader import get_loader

from ops.loss.yolo_loss import YoloLossV7
from ops.metric.DetectionMetric import fitness
from utils.history_collect import History, AverageMeter
from utils.torch_utils import smart_optimizer, smart_resume, smart_scheduler, ModelEMA, de_parallel
from utils.logging import print_args, LOGGER
from utils.lr_warmup import WarmupLR
import val as validate  # for end-of-epoch mAP

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format


# hyp: hyper parameter
# opt: options
def train(model, train_loader, val_loader, device, hyp, opt, names):
    batch_size = opt.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)
    nb = len(train_loader)  # number of batches
    warmup_iter = hyp['warmup_epochs'] * nb

    # ---------- 梯度优化器 ----------
    optimizer = smart_optimizer(model,
                                opt.optimizer,
                                hyp['lr'],
                                hyp['momentum'],
                                hyp['weight_decay'])

    # ---------- 梯度缩放器 ----------
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ---------- 模型参数平滑器 ----------
    ema = ModelEMA(model)

    # ---------- 模型权重加载器 ----------
    best_fitness, last_iter, last_epoch, start_epoch, end_epoch = smart_resume(model,
                                                                               optimizer,
                                                                               ema,
                                                                               opt.epochs,
                                                                               opt.resume,
                                                                               Path(opt.weights))

    # ---------- 学习率优化器 ----------
    scheduler = smart_scheduler(optimizer,
                                opt.scheduler,
                                last_epoch,
                                T_max=end_epoch)

    # ---------- 学习率预热 ----------
    warmer = WarmupLR(optimizer,
                      scheduler,
                      last_iter=last_iter,
                      epoch=end_epoch,
                      momentum=hyp['momentum'],
                      warmup_bias_lr=hyp['warmup_bias_lr'],
                      warmup_iter=warmup_iter,
                      warmup_momentum=hyp['warmup_momentum'])

    # ---------- 记录工具 ----------
    history = History(project_dir=Path(opt.project),
                      name=opt.name,
                      mode='train',
                      save_period=opt.save_period,
                      best_fitness=best_fitness,
                      yaml_args={'hyp': hyp, 'opt': vars(opt)})

    criterion = YoloLossV7(model)

    for epoch in range(start_epoch, end_epoch):
        model.train()

        lbox, lobj, lcls = AverageMeter(), AverageMeter(), AverageMeter()

        LOGGER.info(
            ("\n" + "%11s" * 7) %
            ("Epoch", "GPU_mem", "Size", "box_loss", "obj_loss", "cls_loss", "lr")
        )

        stream = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()
        for i, (images, targets, shape) in enumerate(stream):
            warmer.step()

            images = images.to(device) / 255.

            preds = model(images)

            image_size = torch.as_tensor(shape, device=device)
            loss, loss_items = criterion(preds, targets.to(device), image_size)

            scaler.scale(loss).backward()

            # ------------- 梯度累积 -------------
            if (i + 1) % accumulate == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            lbox.update(loss_items[0].item())
            lobj.update(loss_items[1].item())
            lcls.update(loss_items[2].item())

            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            lr = optimizer.param_groups[0]['lr']
            stream.set_description(
                ("%11i" + "%11s" * 2 + "%11.4g" * 4) %
                (epoch, mem, f"{shape[0]}x{shape[1]}", lbox.avg, lobj.avg, lcls.avg, lr)
            )

        val_metric = validate.run(val_loader=val_loader,
                                  names=names,
                                  model=ema.ema,
                                  history=history,
                                  device=device,
                                  plots=False,
                                  criterion=criterion)

        scheduler.step()

        fi = fitness(np.array(val_metric).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]

        history.save(model, ema, optimizer, epoch, warmer.last_iter, fi)


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./logs/train/exp1/weights/last.pt',
                        help="resume most recent training")
    parser.add_argument("--cfg", type=str, default="./models/yolov7l.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./config/hyp-yolo-v7-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=True, help="resume most recent training")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--optimizer",
                        type=str,
                        choices=["SGD", "Adam", "AdamW"],
                        default="SGD",
                        help="optimizer")
    parser.add_argument("--scheduler",
                        type=str,
                        choices=["Cosine", "MultiStep", "Polynomial", "OneCycleLR"],
                        default="Cosine",
                        help="scheduler")
    parser.add_argument("--workers", type=int, default=3, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./logs", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--save-period", type=int, default=5,
                        help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Automatic DDP Multi-GPU argument, do not modify")

    return parser.parse_args()


def main(opt):
    hyp = OmegaConf.load(Path(opt.hyp))
    cfg = OmegaConf.load(Path(opt.cfg))

    train_loader, val_loader, names = get_loader(hyp, opt)

    device = opt.device
    model = get_model(cfg)
    model.to(device)

    m = de_parallel(model).head  # detection head model
    nl = m.nl  # number of detection layers (to scale hyp)
    nc = m.num_classes
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (max(opt.image_size[0], opt.image_size[1]) / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.hyp = hyp

    train(model, train_loader, val_loader, device, hyp, opt, names)


if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    main(opt)
