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
