from rail1.datasets.cifar10 import load_cifar10
from rail1.datasets.mnist import load_mnist

def cifar10(*args, **kwargs):
    return load_cifar10(*args, **kwargs)


def mnist(*args, **kwargs):
    return load_mnist(*args, **kwargs)