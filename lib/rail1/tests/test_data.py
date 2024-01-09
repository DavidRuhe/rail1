import unittest
from rail1.data import batchloader
import random

# Dummy dataset and collate function for testing
def dummy_dataset(size):
    return list(range(size))

def dummy_collate_fn(batch):
    return batch

class TestBatchLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_size = 100
        self.dataset = dummy_dataset(self.dataset_size)
        self.collate_fn = dummy_collate_fn

    def test_initialization(self):
        """Test the initialization of the BatchLoader."""
        batch_size = 10
        loader = batchloader.BatchLoader(self.dataset, self.collate_fn, batch_size=batch_size)
        self.assertEqual(loader.batch_size, batch_size)

    def test_non_shuffled_batch_retrieval(self):
        """Test retrieving batches without shuffling."""
        batch_size = 5
        loader = batchloader.BatchLoader(self.dataset, self.collate_fn, shuffle=False, batch_size=batch_size)
        batch = loader[0]
        self.assertEqual(batch, [0, 1, 2, 3, 4])

    def test_shuffled_batch_retrieval(self):
        """Test retrieving batches with shuffling."""
        batch_size = 5
        loader = batchloader.BatchLoader(self.dataset, self.collate_fn, shuffle=True, batch_size=batch_size, base_seed=42)
        batch = loader[0]
        random.seed(42)
        expected_batch = [random.randint(0, len(self.dataset) - 1) for _ in range(batch_size)]
        self.assertEqual(batch, expected_batch)


if __name__ == '__main__':
    unittest.main()
