import os
from torchvision import datasets, transforms
from torch.utils import data

DATAROOT = os.environ["DATAROOT"]


def load_cifar10():
    cifar10_train = datasets.CIFAR10(root=DATAROOT, train=True, download=True, transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(root=DATAROOT, train=False, download=True, transform=transforms.ToTensor())
    train_loader = data.DataLoader(cifar10_train, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(cifar10_test, batch_size=128, shuffle=False)
    return {"train_loader": train_loader, "test_loader": test_loader, "val_loader": test_loader}
