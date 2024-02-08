import torch
import os
from torchvision import datasets, transforms

from rail1.data import batchloader
from rail1.data import collate

DATAROOT = os.environ["DATAROOT"]


def load_mnist(batch_size=128, num_workers=0, num_prefetch=0):
    mnist_train = datasets.MNIST(
        root=DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_test = datasets.MNIST(
        root=DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = batchloader.BatchLoader(
        mnist_train,
        collate_fn=collate.default_collate,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=True,
    )
    test_loader = batchloader.BatchLoader(
        mnist_train,
        collate_fn=collate.default_collate,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": test_loader,
    }
