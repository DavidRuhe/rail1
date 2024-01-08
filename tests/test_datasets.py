import os
import unittest
from rail1.datasets import load_cifar10


# class TestLoadCifar10WithoutMocking(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         # Set up an environment variable for DATAROOT
#         # This should be a valid path where the test can download and store the CIFAR10 dataset
#         os.environ["DATAROOT"] = "/path/to/dataroot"

#     def test_load_cifar10(self):
#         # Call the function under test
#         loaders = load_cifar10()

#         # Check that the returned loaders are not None
#         self.assertIsNotNone(loaders.get("train_loader"))
#         self.assertIsNotNone(loaders.get("test_loader"))
#         self.assertIsNotNone(loaders.get("val_loader"))


if __name__ == "__main__":
    unittest.main()
