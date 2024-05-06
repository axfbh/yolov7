import os
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, distributed

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ops.dataset.voc_dataset import VOCDetection
from ops.dataset.utils import detect_collate_fn
import ops.cv.io as io
from ops.transform.resize_maker import ResizeLongestPaddingShort, ResizeShortLongest
from ops.utils.logging import LOGGER, colorstr
from ops.utils.torch_utils import torch_distributed_zero_first
import random

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MyDataSet(VOCDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, bbox_params, classes = super().__getitem__(item)

        resize_sample = ResizeLongestPaddingShort(self.image_size, shuffle=False)(image=image, bbox_params=bbox_params)

        # resize_sample = ResizeShortLongest(self.image_size)(image, bbox_params=bbox_params)

        # io.visualize(resize_sample['image'], resize_sample['bbox_params'], classes, self.id2name)

        if self.augment:
            sample = self.transform(image=resize_sample['image'], bboxes=resize_sample['bbox_params'], classes=classes)
        else:
            sample = {
                'image': resize_sample['image'],
                'bboxes': resize_sample['bbox_params'],
                'classes': classes
            }

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


def create_dataloader(path,
                      image_size,
                      batch_size,
                      names,
                      image_set=None,
                      hyp=None,
                      augment=False,
                      local_rank=-1,
                      workers=3,
                      shuffle=False,
                      seed=0):
    transform = A.Compose([
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

    if augment:
        LOGGER.info(f"{colorstr('albumentations: ')}" + ", ".join(
            f"{x}".replace("always_apply=False, ", "") for x in transform if x.p))

    dataset = MyDataSet(path,
                        image_set=image_set,
                        image_size=image_size,
                        class_name=names,
                        augment=augment,
                        transform=transform if augment else None)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if local_rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle and sampler is None,
                      num_workers=nw,
                      sampler=sampler,
                      pin_memory=PIN_MEMORY,
                      collate_fn=detect_collate_fn,
                      worker_init_fn=seed_worker,
                      generator=generator)
