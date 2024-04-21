from model.modeling import get_model
from data_loader import get_loader
import torch
from utils.epoch_utils import train_epoch, val_epoch
from utils.history_collect import save_model, HistoryLoss
from omegaconf import OmegaConf
import os
from utils.model_freeze import FreezeLayer
from ops.loss.yolo_loss import YoloLossV7
from utils.lr_warmup import WarmupMultiStepLR, WarmupCosineLR
from utils.weight_inject import WeightInject


def setup(args):
    model = get_model(args)
    model.to(args.train.device)
    return model


def train(model, train_loader, val_loader, args):
    batch_size = args.train.batch_size
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    # -------- 梯度优化器 --------
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.solver.lr,
                                momentum=args.sgd.momentum,
                                weight_decay=args.solver.weight_decay)

    # -------- 模型权重加载器 --------
    injector = WeightInject(model=model,
                            optimizer=optimizer,
                            params=args.model.weights,
                            device=args.weight.device)

    injector.load_state_dict("pretrained", "bkpretrained")

    last_epoch = injector.last_epoch
    start_epoch = last_epoch + 1
    end_epoch = args.iter_max + 1

    # -------- 学习率优化器 and 学习率预热器 --------
    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=[140, 220],
                                  gamma=0.1,
                                  last_epoch=last_epoch,
                                  warmup_method=args.warmup.warmup_method,
                                  warmup_factor=args.warmup.warmup_factor,
                                  warmup_iters=args.warmup.warmup_iters)

    # scheduler = WarmupCosineLR(optimizer,
    #                            end_epoch,
    #                            last_epoch,
    #                            warmup_method=args.warmup.warmup_method,
    #                            warmup_factor=args.warmup.warmup_factor,
    #                            warmup_iters=args.warmup.warmup_iters)

    # -------- 每个 epoch 的 loss 记录工具 --------
    history_loss = HistoryLoss(args.log_info.path,
                               args.log_info.num,
                               modes=['train', 'val'])

    for epoch in range(start_epoch, end_epoch):
        train_metric = train_epoch(model=model,
                                   loader=train_loader,
                                   device=args.train.device,
                                   epoch=epoch,
                                   optimizer=optimizer,
                                   criterion=YoloLossV7(args, g=1, thresh=4),
                                   accumulate=accumulate)

        # val_metric = val_epoch(model=model,
        #                        loader=val_loader,
        #                        device=args.train.device,
        #                        epoch=epoch,
        #                        criterion=YoloLossV3(args, 0.296))

        scheduler.step()

        save_model(model, optimizer, train_metric)
        history_loss.append(train_metric['loss'].avg,
                            0)
        history_loss.loss_plot(start=start_epoch)


def main():
    args = OmegaConf.load('./config/config.yaml')

    model = setup(args)

    train_loader, val_loader = get_loader(args)

    train(model, train_loader, val_loader, args)


if __name__ == '__main__':
    print('---------------')
    main()
