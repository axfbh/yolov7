import torch
from tqdm import tqdm
from utils.history_collect import AverageMeter
from ops.metric.DetectionMetric import process_batch, ap_per_class
from torchvision.ops.boxes import box_convert
from ops.detection.nms import non_max_suppression
import numpy as np
from utils.logging import LOGGER


def train_epoch(model, loader, device, epoch, optimizer, criterion, scaler, accumulate=1):
    model.train()

    metric = {
        'epoch': epoch,
        'lbox': AverageMeter(),
        'lobj': AverageMeter(),
        'lcls': AverageMeter(),
    }

    LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "Size", "box_loss", "obj_loss", "cls_loss", "lr"))

    stream = tqdm(loader, bar_format="{l_bar}{bar:10}{r_bar}")

    optimizer.zero_grad()
    for i, (images, targets) in enumerate(stream):
        images = images.to(device) / 255.

        _, _, h, w = images.size()

        image_size = torch.tensor([h, w])

        preds = model(images)

        loss, lbox, lobj, lcls = criterion(preds, targets.to(device), image_size.to(device))

        scaler.scale(loss).backward()

        # ------------- 梯度累积 -------------
        if (i + 1) % accumulate == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

        metric['lbox'].update(lbox.item())
        metric['lobj'].update(lobj.item())
        metric['lcls'].update(lcls.item())

        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        lr = optimizer.param_groups[0]['lr']
        stream.set_description(
            ("%11s" * 3 + "%11.4g" * 4)
            % (str(epoch), mem, str(h) + 'x' + str(w), metric['lbox'].avg, metric['lobj'].avg, metric['lcls'].avg, lr)
        )

    return metric


@torch.no_grad()
def val_epoch(model, loader, device, epoch, criterion):
    model.eval()

    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")

    stream = tqdm(loader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}")

    metric = {
        'epoch': epoch,
    }

    stats, ap, ap_class = [], [], []

    seen = 0
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    single_cls = False

    for i, data in enumerate(stream):
        images, targets = data

        image_size = torch.as_tensor(images.shape[2:])

        preds, train_out = model(images.to(device))

        _, lbox, lobj, lcls = criterion(train_out, targets.to(device), image_size.to(device))

        preds = non_max_suppression(preds, 0.001, 0.6)

        for si, pred in enumerate(preds):
            labels = targets[si, :, 1:]
            labels = labels[labels[:, 0] > 0].to(device)
            labels[:, 0] = labels[:, 0] - 1
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            if single_cls:
                pred[:, 5] = 0

            predn = pred.clone()

            if nl:
                tbox = box_convert(labels[:, 1:5], 'cxcywh', 'xyxy')  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=20)  # number of targets per class

    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    metric['map50'] = map50
    metric['map'] = map
    metric['mp'] = mp
    metric['mr'] = mr

    return metric
