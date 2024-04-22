import os

import torch
from torch.utils.data import DataLoader
from ops.dataset.voc_dataset import VOCDetection, CLASSES_NAME
from ops.dataset.utils import detect_collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import ops.cv.io as io
from ops.transform.resize_maker import ResizeLongestPaddingShort

np.random.seed(0)


class MyDataSet(VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, bbox_params, classes = super().__getitem__(item)

        # io.visualize(image, bbox_params, classes, CLASSES_NAME)

        resize_sample = ResizeLongestPaddingShort(self.image_size, shuffle=False)(image=image, bbox_params=bbox_params)

        sample = self.transform(image=resize_sample['image'], bboxes=resize_sample['bbox_params'], classes=classes)

        image = ToTensorV2()(image=sample['image'])['image']
        bbox_params = torch.FloatTensor(sample['bboxes'])
        classes = torch.LongTensor(sample['classes'])[:, None]

        gxy = (bbox_params[:, 2:] + bbox_params[:, :2]) * 0.5
        gwy = bbox_params[:, 2:] - bbox_params[:, :2]

        indices = torch.zeros_like(classes)

        target = torch.hstack([
            indices,
            classes,
            gxy,
            gwy
        ])

        return image / 255., target


def get_loader(args):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=args.image_size[0],
                            width=args.image_size[1],
                            scale=(0.8, 1.0),
                            ratio=(0.9, 1.11),
                            p=0.0),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.HueSaturationValue(),
        A.Affine(scale=0.1, shear=10, rotate=90, cval=(114, 114, 114)),
    ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    val_transform = A.Compose([
    ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    train_dataset = MyDataSet(args.train.dataset.path,
                              'car_train',
                              args,
                              train_transform)

    val_dataset = MyDataSet(args.val.dataset.path,
                            'car_val',
                            args,
                            val_transform)

    nw = min(3, args.train.batch_size)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train.batch_size,
                              shuffle=False,
                              collate_fn=detect_collate_fn,
                              persistent_workers=True,
                              num_workers=nw,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val.batch_size,
                            shuffle=False,
                            collate_fn=detect_collate_fn,
                            persistent_workers=True,
                            num_workers=nw,
                            drop_last=True)

    return train_loader, val_loader
