import os.path
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torchvision.ops.boxes import box_convert
from ops.metric.DetectionMetric import process_batch, ap_per_class

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.modeling import get_model
from ops.detection.nms import non_max_suppression
from data_loader import get_loader

from utils.logging import print_args, LOGGER
from utils.history_collect import History
from utils.plots import plot_images, output_to_target


@torch.no_grad()
def run(val_loader,
        names,
        model,
        history,
        device,
        conf_thres=0.001,
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        plots=True,
        compute_loss=None):
    model.eval()

    metric = {}

    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    stream = tqdm(val_loader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}")

    seen = 0
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    single_cls = False

    for batch_i, (images, targets) in enumerate(stream):
        images = images.to(device, non_blocking=True) / 255.
        targets = targets.to(device)

        preds, train_out = model(images)

        if compute_loss:
            image_size = torch.as_tensor(images.shape[2:]).to(device)
            loss += compute_loss(train_out, targets, image_size)[1]  # box, obj, cls

        preds = non_max_suppression(preds, conf_thres, iou_thres, max_det)

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

        if plots and batch_i < 3:
            plot_images(images, targets, names)  # labels
            plot_images(images, output_to_target(preds), names)  # pred

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


def parse_opt():
    parser = argparse.ArgumentParser()
    # -------------- 参数文件 --------------
    parser.add_argument("--cfg", type=str, default="./models/yolov7l.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default="./data/voc.yaml", help="dataset.yaml path")

    # -------------- 参数值 --------------
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--image-size", type=list, default=[640, 640], help="train, val image size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--resume", default='./logs/train/exp1/weights/last.pt', help="resume most recent training")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    parser.add_argument("--workers", type=int, default=3, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="./logs", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")

    return parser.parse_args()


def main():
    opt = parse_opt()
    print_args(vars(opt))

    hyp = OmegaConf.load(Path(opt.hyp))
    cfg = OmegaConf.load(Path(opt.cfg))
    data = OmegaConf.load(Path(opt.data))

    device = opt.device
    model = get_model(cfg)
    model.to(device)

    train_loader, val_loader, names = get_loader(hyp, opt)

    history = History(project_dir=Path(opt.project),
                      name=opt.name,
                      mode='val',
                      save_period=opt.save_period)

    run(val_loader, names, model, history, device)


if __name__ == '__main__':
    main()
