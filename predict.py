import os.path

import numpy as np

from model.modeling import get_model
import torch

import cv2
from omegaconf import OmegaConf
import albumentations as A
from ops.detection.nms import non_max_suppression
from albumentations.pytorch import ToTensorV2
from ops.transform.resize_maker import ResizeLongestPaddingShort
import ops.cv.io as io
from ops.dataset.utils import batch_images
from ops.dataset.voc_dataset import id2name
from ops.detection.postprocess_utils import YoloPostProcess


def setup(args):
    model = get_model(args)
    if os.path.exists(args.model.weights.checkpoints):
        print('loading checkpoint')
        model.load_state_dict(torch.load(args.model.weights.checkpoints)['model'])
    else:
        raise IndexError(f'cannot find the {args.model.weights.checkpoints}')
    model.to(args.test.device)
    return model


def inference(model, image, device):
    preds, _ = model(image.to(device))

    # ------- 补充 ---------
    return non_max_suppression(preds, 0.3, 0.4)
s

@torch.no_grad()
def predict(model, args):
    model.eval()

    root = args.test.dataset.root

    device = args.test.device

    with open(args.test.dataset.path, 'r') as fp:
        loader = fp.readlines()

    batch_image = []
    images = []
    for i in range(len(loader)):
        image_path = os.path.join(root, loader[i].strip().split(' ')[0])
        image = io.imread(image_path)

        pad_sample = ResizeLongestPaddingShort(args.image_size, shuffle=False)(image)

        tensor_image = ToTensorV2()(image=pad_sample['image'])['image'] / 255.

        batch_image.append(tensor_image)
        images.append(pad_sample['image'].copy())

        if (i + 1) % 5 == 0:
            batch_image = batch_images(batch_image)

            output = inference(model, batch_image, device)

            for i, out in enumerate(output):
                if out is not None:
                    for p in out:
                        cv2.rectangle(images[i],
                                      tuple(p[:2].cpu().int().numpy().copy()),
                                      tuple(p[2:4].cpu().int().numpy().copy()),
                                      (0, 255, 0), 1)
                        cv2.putText(images[i],
                                    id2name.get(p[5].item() + 1),
                                    tuple(p[:2].cpu().int().numpy().copy() + np.array([-5, -5])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 0, 255), 2)
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
                io.show('nam', images[i])
            batch_image = []
            images.clear()


def main():
    args = OmegaConf.load("./config/config.yaml")

    model = setup(args)

    predict(model, args)


if __name__ == '__main__':
    main()
