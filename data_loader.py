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

        image = sample['image']
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
        A.HorizontalFlip(p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        ], p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ToTensorV2()
    ], A.BboxParams(format='pascal_voc', label_fields=['classes']))

    train_dataset = MyDataSet(args.train.dataset.path,
                              'car_train',
                              args,
                              train_transform)

    nw = min(3, args.train.batch_size, os.cpu_count() // 2 - 6)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train.batch_size,
                              shuffle=False,
                              collate_fn=detect_collate_fn,
                              persistent_workers=True,
                              num_workers=nw,
                              drop_last=True)

    return train_loader, 0
