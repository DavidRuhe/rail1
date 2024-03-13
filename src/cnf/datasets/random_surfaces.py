import torch
import torch.utils.data
import os
import sys
import numpy as np
from rail1.data import batchloader

DATAROOT = os.environ["DATAROOT"]


def sample_sphere(npoints, r=1, ndim=3):
    vec = torch.randn(npoints, ndim)
    vec *= r / torch.linalg.norm(vec, axis=1, keepdim=True)
    return vec


class RandomSurfaces(torch.utils.data.Dataset):

    def __init__(self, n_points_per_shape=2048):
        self.n_points_per_shape = n_points_per_shape

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        return sample_sphere(self.n_points_per_shape // 2)


def load_random_surface_dataset(
    n_points_per_shape=2048,
    batch_size=128,
    num_workers=0,
    num_prefetch=0,
):

    train = RandomSurfaces(n_points_per_shape=n_points_per_shape)
    test = RandomSurfaces(n_points_per_shape=n_points_per_shape)

    train_loader = batchloader.BatchLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=True,
    )

    test_loader = batchloader.BatchLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": test_loader,
    }


# def load_random_points_dataset(
#     n_points=1, dim=3, batch_size=128, num_workers=0, num_prefetch=0, return_basis=True,
# ):

#     train = RandomPoints(n_points=n_points, dim=dim, return_basis=return_basis)
#     train_loader = batchloader.BatchLoader(
#         train,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         n_prefetch=num_prefetch,
#         shuffle=True,
#     )
#     return {
#         "train_loader": train_loader,
#         "val_loader": None,
#         "test_loader": None,
#     }
