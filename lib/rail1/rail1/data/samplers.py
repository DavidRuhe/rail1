import random  # pragma: no cover
import torch  # pragma: no cover
from torch.utils.data import Sampler  # pragma: no cover


class InfiniteRandomSampler(Sampler):  # pragma: no cover
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            index = random.randint(0, len(self.data_source) - 1)  # type: ignore
            yield index

    def __len__(self):
        return torch.iinfo(torch.int64).max


class SequentialSampler(Sampler):  # pragma: no cover
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)