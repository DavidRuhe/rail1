import sys
import torch
import torch.utils.data
import os
from torchvision import datasets, transforms

from rail1.data import batchloader
from rail1.data import collate

DATAROOT = os.environ["DATAROOT"]

EPS = 0.1


class GravitationalField(torch.utils.data.Dataset):

    def __init__(self, num_points=8, length=sys.maxsize):
        self.num_points = num_points
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        O = torch.rand(self.num_points, 2) * 2 - 1
        O[:, None] - O[:, :, None]
        Z_O = (
            (O[:, None, 0] - O[None, :, 0]) ** 2
            + (O[:, None, 1] - O[None, :, 1]) ** 2
            + EPS**2
        ) ** (3 / 2)

        F_x_O = -torch.sum((O[:, None, 0] - O[None, :, 0]) / Z_O, dim=-1)
        F_y_O = -torch.sum((O[:, None, 1] - O[None, :, 1]) / Z_O, dim=-1)

        F_O = torch.stack([F_x_O, F_y_O], dim=-1)
        F_O_mag = torch.norm(F_O, dim=-1)
        return O, F_O, F_O_mag


def load_gravitational_field(batch_size=128, num_workers=0, num_prefetch=0):

    gf = GravitationalField(sys.maxsize)

    train_loader = batchloader.BatchLoader(
        gf,
        collate_fn=collate.default_collate,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "test_loader": train_loader,
        "val_loader": train_loader,
        "train_dataset": gf,
        "test_dataset": gf,
    }
