from .autoencoder import Autoencoder


def mnist_autoencoder():
    return Autoencoder(784, (512, 256, 128))
