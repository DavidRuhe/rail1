import unittest
from rail1.datasets import cifar10

class TestDatasets(unittest.TestCase):
    def test_cifar10(self):
        datasets = cifar10()