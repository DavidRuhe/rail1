import unittest
from torch import nn
from rail1 import optimizers

class TestOptimizers(unittest.TestCase):

    def test_adam(self):

        model = nn.Linear(1, 1)
        adam = optimizers.adam(model)

        