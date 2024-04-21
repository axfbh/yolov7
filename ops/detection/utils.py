import torch


def make_grid(h, w, sh, sw, device):
    shifts_x = torch.arange(0, w, dtype=torch.int32, device=device) * sw
    shifts_y = torch.arange(0, h, dtype=torch.int32, device=device) * sh

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y), dim=1)
    return shifts
