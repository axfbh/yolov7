import os
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.dataset.voc_dataset import VOCDetection
from ops.dataset.utils import detect_collate_fn
import ops.cv.io as io
from ops.transform.resize_maker import ResizeLongestPaddingShort, ResizeShortLongest
from ops.utils.logging import LOGGER, colorstr

np.random.seed(0)


class MyDataSet(VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, bbox_params, classes = super().__getitem__(item)

        resize_sample = ResizeLongestPaddingShort(self.image_size, shuffle=False)(image=image, bbox_params=bbox_params)

        # resize_sample = ResizeShortLongest(self.image_size)(image, bbox_params=bbox_params)

        # io.visualize(resize_sample['image'], resize_sample['bbox_params'], classes, self.id2name)

        sample = self.transform(image=resize_sample['image'], bboxes=resize_sample['bbox_params'], classes=classes)

        image = ToTensorV2()(image=sample['image'])['image'].float()
        bbox_params = torch.FloatTensor(sample['bboxes'])
        classes = torch.LongTensor(sample['classes'])[:, None]

        nl = len(bbox_params)

        if nl:
            gxy = (bbox_params[:, 2:] + bbox_params[:, :2]) * 0.5
            gwy = bbox_params[:, 2:] - bbox_params[:, :2]
        else:
            gxy = torch.zeros((nl, 2))
            gwy = torch.zeros((nl, 2))

        indices = torch.zeros_like(classes)

        target = torch.hstack([
            indices,
            classes,
            gxy,
            gwy
        ])

        return image, target


def get_loader(hyp, opt):
    train_transform = A.Compose([
        A.Affine(scale={"x": (1 - hyp.scale, 1 + hyp.scale),
                        "y": (1 - hyp.scale, 1 + hyp.scale)},
                 translate_percent={"x": (0.5 - hyp.translate, 0.5 + hyp.translate),
                                    "y": (0.5 - hyp.translate, 0.5 + hyp.translate)},
                 cval=114,
                 p=0.8),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HueSaturationValue(p=0.8),
        A.HorizontalFlip(p=hyp.fliplr),
        A.VerticalFlip(p=hyp.flipud),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes']))
    LOGGER.info(f"{colorstr('albumentations: ')}" + ", ".join(
        f"{x}".replace("always_apply=False, ", "") for x in train_transform if x.p))

    val_transform = A.Compose([
    ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    data = OmegaConf.load(Path(opt.data))

    train_dataset = MyDataSet(Path(data.train),
                              image_set='car_train',
                              image_size=opt.image_size,
                              class_name=data.names,
                              transform=train_transform)

    val_dataset = MyDataSet(Path(data.val),
                            image_set='car_val',
                            image_size=opt.image_size,
                            class_name=data.names,
                            transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              collate_fn=detect_collate_fn,
                              persistent_workers=True,
                              num_workers=opt.workers,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            collate_fn=detect_collate_fn,
                            persistent_workers=True,
                            num_workers=opt.workers,
                            drop_last=True)

    return train_loader, val_loader, data.names
