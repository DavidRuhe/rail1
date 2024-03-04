import torch
import torch.utils.data
import os
from torchvision import datasets, transforms

from rail1.data import batchloader
from rail1.data import collate

DATAROOT = os.environ["DATAROOT"]


class EmbeddedMNIST(torch.utils.data.Dataset):

    def __init__(self, embedding_dir, train=False):
        self.embedding_dir = embedding_dir

        if train:
            self.mnist = datasets.MNIST(
                root=DATAROOT,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            self.embedding = torch.load(os.path.join(embedding_dir, "train.pt"))
        else:
            self.mnist = datasets.MNIST(
                root=DATAROOT,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            self.embedding = torch.load(os.path.join(embedding_dir, "test.pt"))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        return *self.mnist[idx], self.embedding[idx]


def load_embedded_mnist(
    embedding_dir, batch_size=128, num_workers=0, num_prefetch=0, subset=None
):

    mnist_train = EmbeddedMNIST(embedding_dir, train=True)

    if subset is not None:
        mnist_train = torch.utils.data.Subset(mnist_train, subset)

    mnist_test = EmbeddedMNIST(embedding_dir, train=False)

    train_loader = batchloader.BatchLoader(
        mnist_train,
        collate_fn=collate.default_collate,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=num_prefetch,
        shuffle=True,
    )
    test_loader = batchloader.BatchLoader(
        mnist_test,
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
        "train_dataset": mnist_train,
        "test_dataset": mnist_test,
    }
