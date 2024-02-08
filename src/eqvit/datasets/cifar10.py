import torch
import os
from torchvision import transforms, datasets
import torch.utils.data

DATAROOT = os.environ["DATAROOT"]
CIFAR10_NORMALIZATION = [0.49139968, 0.48215841, 0.44653091], [
    0.24703223,
    0.24348513,
    0.26158784,
]


def get_cifar10():
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*CIFAR10_NORMALIZATION)]
    )

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR10_NORMALIZATION),
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = datasets.CIFAR10(
        root=DATAROOT, train=True, transform=train_transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=DATAROOT, train=False, transform=test_transform, download=True
    )
    return train_dataset, test_dataset
