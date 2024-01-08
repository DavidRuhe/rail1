import random
import multiprocessing
import collections
import queue


class BatchLoader:
    def __init__(self, dataset, collate_fn, batch_size=1, base_seed=0, num_workers=0, n_prefetch=0, timeout=32):
        self.base_seed = base_seed
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.n_prefetch = n_prefetch
        self.timeout=timeout

        if self.num_workers <= 0:
            assert n_prefetch <= 0

        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        self.batch_buffer = collections.defaultdict(list)
        self.workers = [multiprocessing.Process(target=self.worker_task) for _ in range(num_workers)]
        for worker in self.workers:
            worker.start()

    def worker_task(self):
        while True:
            task = self.input_queue.get()
            if task is None:  
                break
            batch_index, local_index, data_index = task
            self.output_queue.put((batch_index, local_index, self.dataset[data_index]))
    
    def __getitem__(self, index):
        if self.num_workers <= 0:
            rng = random.Random(self.base_seed + index)
            return self.collate_fn([self.dataset[rng.randint(0, len(self.dataset) - 1)] for _ in range(self.batch_size)])
        else:
            for batch_index in range(index, index + self.n_prefetch):
                if batch_index in self.batch_buffer:
                    continue
                rng = random.Random(self.base_seed + batch_index)
                for local_index in range(self.batch_size):
                    self.input_queue.put((batch_index, local_index, rng.randint(0, len(self.dataset) - 1)))

            while len(self.batch_buffer[index]) < self.batch_size:
                try:
                    batch_index, local_index, data = self.output_queue.get(timeout=self.timeout)
                except queue.Empty:
                    print("Dataloader timeout, closing...")
                    return self.close()
                self.batch_buffer[batch_index].append((local_index, data))

            batch = self.batch_buffer.pop(index)
            # Sort
            batch.sort(key=lambda x: x[0])
            return self.collate_fn([x[1] for x in batch])


    def close(self):
        if self.num_workers > 0:
            for _ in range(self.num_workers):
                self.input_queue.put(None)
            for p in self.workers:
                p.join()





