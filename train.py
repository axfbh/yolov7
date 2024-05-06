import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import numpy as np

from models.modeling import get_model
from dataloader import create_dataloader
import val as validate  # for end-of-epoch mAP

from ops.loss.yolo_loss import YoloLossV7, YoloLossV4, YoloLossV5
from ops.loss.fcos_loss import FcosLoss
from ops.metric.DetectionMetric import fitness
from ops.utils.history_collect import History, AverageMeter
from ops.utils.torch_utils import smart_optimizer, smart_resume, smart_scheduler, ModelEMA, de_parallel, select_device, \
    init_seeds
from ops.utils.logging import print_args, LOGGER, colorstr
from ops.utils.lr_warmup import WarmupLR

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


# hyp: hyper parameter
# opt: options
def train(model, train_loader, val_loader, device, hyp, opt, names):
    batch_size = opt.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)
    nb = len(train_loader)  # number of batches
    warmup_iter = max(round(hyp["warmup_epochs"] * nb), 100)
    hyp["weight_decay"] *= batch_size * accumulate / nbs
    cuda = device.type != "cpu"

    # ---------- 梯度优化器 ----------
    optimizer = smart_optimizer(model,
                                opt.optimizer,
                                hyp['lr'],
                                hyp['momentum'],
                                hyp['weight_decay'])

    # ---------- 梯度缩放器 ----------
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)

    # ---------- 模型参数平滑器 ----------
    ema = ModelEMA(model) if RANK in {-1, 0} else None

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

    # ---------- 学习率预热器 ----------
    warmer = WarmupLR(optimizer,
                      scheduler,
                      last_iter=last_iter,
                      momentum=hyp['momentum'],
                      warmup_bias_lr=hyp['warmup_bias_lr'],
                      warmup_iter=warmup_iter,
                      warmup_momentum=hyp['warmup_momentum'])

    # ---------- 记录工具 ----------
    if RANK in {-1, 0}:
        history = History(project_dir=Path(opt.project),
                          name=opt.name,
                          mode='train',
                          save_period=opt.save_period,
                          best_fitness=best_fitness,
                          yaml_args={'hyp': hyp, 'opt': vars(opt)})

    # DDP mode，
    # torch.distributed.run 启动
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # criterion = FcosLoss(model)
    criterion = YoloLossV5(model)

    for epoch in range(start_epoch, end_epoch):
        model.train()

        mloss = AverageMeter()

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)

        if RANK in {-1, 0}:
            LOGGER.info(
                ("\n" + "%11s" * 7) %
                ("Epoch", "GPU_mem", "Size", "box_loss", "obj_loss", "cls_loss", "lr")
            )
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()
        for i, (images, targets, shape) in pbar:
            warmer.step()

            images = images.to(device) / 255.

            preds = model(images)

            image_size = torch.as_tensor(shape, device=device)
            loss, loss_items = criterion(preds, targets.to(device), image_size)

            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            scaler.scale(loss).backward()

            # ------------- 梯度累积 -------------
            if (warmer.last_iter + 1) % accumulate == 0:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if RANK in {-1, 0}:
                mloss += loss_items
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                lr = optimizer.param_groups[0]['lr']
                pbar.set_description(
                    ("%11i" + "%11s" * 2 + "%11.4g" * 4) %
                    (epoch, mem, f"{shape[0]}x{shape[1]}", *mloss, lr)
                )

        scheduler.step()

        if RANK in {-1, 0}:
            val_metric = validate.run(val_loader=val_loader,
                                      names=names,
                                      model=ema.ema,
                                      history=history,
                                      device=device,
                                      plots=False,
                                      single_cls=opt.single_cls,
                                      criterion=criterion)
            fi = fitness(np.array(val_metric).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            history.save(model, ema, optimizer, epoch, warmer.last_iter, fi)

    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--weights", default='./logs/train/exp/weights/last.pt', help="resume most recent training")
    parser.add_argument("--cfg", type=str, default="./models/yolo-v4-v5-m.yaml", help="models.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="./data/hyp/hyp-yolo-v5-low.yaml", help="hyperparameters path")

    # -------------- 参数值 --------------
    parser.add_argument("--epochs", type=int, default=300, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
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
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=1, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./logs", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--save-period", type=int, default=5,
                        help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Automatic DDP Multi-GPU argument, do not modify")

    return parser.parse_args()


def main(rank, world_size, opt):
    hyp = OmegaConf.load(Path(opt.hyp))
    cfg = OmegaConf.load(Path(opt.cfg))
    data = OmegaConf.load(Path(opt.data))

    if RANK in {-1, 0}:
        print_args(vars(opt))
        LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    device = select_device(opt.device, batch_size=opt.batch_size)

    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                rank=rank,
                                world_size=world_size)

    model = get_model(cfg)
    model.to(device)

    m = de_parallel(model).head  # detection head models
    nl = m.nl  # number of detection layers (to scale hyp)
    nc = m.num_classes
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (max(opt.image_size[0], opt.image_size[1]) / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.hyp = hyp

    init_seeds(opt.seed + 1 + RANK)

    names = data.names

    train_loader = create_dataloader(Path(data.train),
                                     opt.image_size,
                                     opt.batch_size // WORLD_SIZE,
                                     names,
                                     hyp=hyp,
                                     image_set='car_train',
                                     augment=True,
                                     local_rank=LOCAL_RANK,
                                     workers=opt.workers,
                                     shuffle=True,
                                     seed=opt.seed)

    val_loader = create_dataloader(Path(data.val),
                                   opt.image_size,
                                   opt.batch_size // WORLD_SIZE * 2,
                                   data.names,
                                   hyp=hyp,
                                   image_set='car_val',
                                   augment=False,
                                   local_rank=LOCAL_RANK,
                                   workers=opt.workers,
                                   shuffle=True,
                                   seed=opt.seed) if RANK in {-1, 0} else None

    train(model, train_loader, val_loader, device, hyp, opt, names)


def init_process(local_rank, node_rank, local_size, world_size, fn):
    global LOCAL_RANK, RANK, WORLD_SIZE
    rank = local_rank + node_rank * local_size

    os.environ['MASTER_ADDR'] = '192.168.2.66'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['LOCAL_RANK'] = str(local_size)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    LOCAL_RANK = local_rank
    RANK = rank
    WORLD_SIZE = world_size

    opt = parse_opt()
    fn(rank, world_size, opt)


if __name__ == '__main__':
    # 总共 GPU 数量
    world_size = 2

    # 当前 机器 GPU 数量
    nproc_per_node = 1

    # 当前 机器 ID
    node_rank = 0

    mp.spawn(init_process,
             args=(node_rank, nproc_per_node, world_size, main),
             nprocs=nproc_per_node,
             join=True)
