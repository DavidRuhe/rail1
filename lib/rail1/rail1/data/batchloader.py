import random
import multiprocessing
from torch.utils.data import dataloader
from rail1.utils import math as math_utils


class BatchLoader:
    def __init__(
        self,
        dataset,
        collate_fn=None,
        shuffle=False,
        batch_size=1,
        base_seed=0,
        num_workers=0,
        n_prefetch=0,
        timeout=128,
    ):
        self.base_seed = base_seed
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = (
            collate_fn if collate_fn is not None else dataloader.default_collate
        )
        self.num_workers = num_workers
        self.n_prefetch = n_prefetch
        self.timeout = timeout
        self.shuffle = shuffle

        if self.num_workers <= 0:
            assert n_prefetch <= 0
            self.input_queue = None
            self.output_queue = None
            self.batch_buffer = None
            self.workers = None
        else:
            self.input_queue = multiprocessing.Queue()
            self.output_queue = multiprocessing.Queue()
            self.batch_buffer = dict()
            self.workers = [
                multiprocessing.Process(target=self.worker_task)
                for _ in range(num_workers)
            ]
            for worker in self.workers:  # pragma: no cover
                worker.start()

        if not shuffle:
            self.list_of_indices = [
                list(range(i, min(i + batch_size, len(dataset))))
                for i in range(0, len(dataset) + 1, batch_size)
            ]
        else:
            self.list_of_indices = None

    def worker_task(self):  # pragma: no cover
        while True:
            task = self.input_queue.get()
            if task is None:
                print("Worker received poison pill, closing...")
                break
            batch_index, local_index, data_index = task
            self.output_queue.put(
                (batch_index, local_index, self.dataset[data_index], data_index)
            )

    def __len__(self):
        if self.shuffle:
            raise ValueError("Length of a shuffled dataset is not well-defined.")
        return math_utils.ceildiv(len(self.dataset), self.batch_size)

    def __getitem__(self, index):
        if self.num_workers <= 0:
            rng = random.Random(self.base_seed + index)
            if self.shuffle:
                indices = [
                    rng.randint(0, len(self.dataset) - 1)
                    for _ in range(self.batch_size)
                ]
            else:
                indices = self.list_of_indices[index]
            return self.collate_fn([self.dataset[i] for i in indices])
        else:  # pragma: no cover
            prefetch_index = index + self.n_prefetch + 1
            if not self.shuffle:
                prefetch_index = min(prefetch_index, len(self.list_of_indices))
            for batch_index in range(index, prefetch_index):
                if batch_index in self.batch_buffer:  # pragma: no cover
                    continue
                self.batch_buffer[batch_index] = dict()
                rng = random.Random(self.base_seed + batch_index)
                if self.shuffle:
                    indices = [
                        rng.randint(0, len(self.dataset) - 1)
                        for _ in range(self.batch_size)
                    ]
                else:
                    indices = self.list_of_indices[batch_index]
                local_indices = list(range(len(indices)))

                for local_index, data_index in zip(local_indices, indices):
                    self.input_queue.put((batch_index, local_index, data_index))

            batch_size = self.batch_size if self.shuffle else len(self.list_of_indices[index])
            while len(self.batch_buffer[index]) < batch_size:

                batch_index, local_index, data, data_index = self.output_queue.get(
                    timeout=self.timeout
                )
                self.batch_buffer[batch_index][local_index] = data

            batch = self.batch_buffer.pop(index)
            return self.collate_fn([batch[k] for k in sorted(batch)])

    def close(self):  # pragma: no cover
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        for worker in self.workers:
            try:
                worker.terminate()
                worker.join()
                worker.close()
            except ValueError:
                pass
