import torch
from typing import List
import torch.nn as nn
from ops.detection.utils import make_grid


class AnchorGenerator(nn.Module):
    def __init__(self, sizes: List, device):
        super(AnchorGenerator, self).__init__()
        self.strides = None
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        self.device = device
        self.cell_anchors = [self.cell_anchor(size) for size in sizes]

    def cell_anchor(self, scale):
        """
        grid [x0,y0,x1,y1] -> [-4,-4,4,4], center of grid is [0,0]
        :param scale:
        :return:
        """
        w, h = scale, scale

        base_anchors = torch.tensor([-w, -h, w, h], device=self.device, dtype=torch.float32) / 2
        return base_anchors.round()

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List):
        anchors = []

        cell_anchors = self.cell_anchors

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            shifts = make_grid(grid_height, grid_width, stride_height, stride_width, self.device).repeat([1, 2])

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def forward(self, image_size, grid_sizes):
        if len(self.sizes) != len(grid_sizes):
            raise ValueError('sizes 和 grid 长度不一')

        strides = [
            [
                image_size[0] // g[0],
                image_size[1] // g[1],
            ]
            for g in grid_sizes
        ]

        self.strides = torch.tensor(strides, dtype=torch.float32, device=self.device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        return anchors_over_all_feature_maps

# #
# if __name__ == '__main__':
#     anchor_generator = AnchorGenerator([8, 16, 32, 64, 128], 'cuda')
#     anchor_generator([800, 1216], [[100, 152], [50, 76], [25, 38], [13, 19], [7, 10]])
