import torch
import torch.utils.data
import os
from torchvision import datasets, transforms
import sys

from rail1.data import batchloader
from rail1.data import collate
import numpy as np

DATAROOT = os.environ["DATAROOT"]


def sample_sphere(npoints, r=1, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec *= r / np.linalg.norm(vec, axis=0)
    return vec

class RandomSurfaces(torch.utils.data.Dataset):

    def __init__(self, n_points_per_shape=2048):
        self.n_points_per_shape = n_points_per_shape

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        positive = sample_sphere(self.n_points_per_shape // 2).T
        negative = np.random.uniform(-1, 1, (self.n_points_per_shape // 2, 3))
        return positive, negative




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


def

    