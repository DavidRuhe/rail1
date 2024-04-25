import torch
import torch.utils.data
import os
from torchvision import datasets, transforms
import sys

from rail1.data import batchloader
from rail1.data import collate

DATAROOT = os.environ["DATAROOT"]


class RandomPoints(torch.utils.data.Dataset):

    def __init__(self, num_points=1, dim=3, return_basis=False):
        self.return_basis = return_basis
        self.num_points = num_points
        self.dim = dim

        if return_basis:
            self.basis = torch.eye(dim)

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        points = torch.randn(self.num_points, self.dim)
        if self.return_basis:
            points = torch.cat([self.basis, points], dim=0)
        return points


def load_random_points_dataset(
    num_points=1, dim=3, batch_size=128, num_workers=0, num_prefetch=0, return_basis=True,
):

    train = RandomPoints(num_points=num_points, dim=dim, return_basis=return_basis)
    train_loader = batchloader.BatchLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=True,
    )
    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": None,
    }
