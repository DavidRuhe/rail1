import torch
from rail1.data import batchloader
from datasets.reverse import get_reverse
from datasets.anomaly import get_anomaly
from datasets.cifar10 import get_cifar10


def reverse(batch_size=128, num_classes=10):
    train_dataset, val_dataset, test_dataset = get_reverse(num_classes=num_classes)

    def collate_fn(batch):
        inp_data, labels = zip(*batch)
        inp_data = torch.stack(inp_data)
        labels = torch.stack(labels)
        return inp_data, labels

    return {
        "train_loader": batchloader.BatchLoader(
            train_dataset, collate_fn, batch_size=batch_size, shuffle=True
        ),
        "val_loader": batchloader.BatchLoader(
            val_dataset, collate_fn, batch_size=batch_size
        ),
        "test_loader": batchloader.BatchLoader(
            test_dataset, collate_fn, batch_size=batch_size
        ),
    }


def anomaly(batch_size=64):
    train_dataset, val_dataset, test_dataset = get_anomaly()

    def collate_fn(batch):
        return tuple(torch.stack(z) for z in zip(*batch))

    return {
        "train_loader": batchloader.BatchLoader(
            train_dataset, collate_fn, batch_size=batch_size, shuffle=True
        ),
        "val_loader": batchloader.BatchLoader(
            val_dataset, collate_fn, batch_size=batch_size
        ),
        "test_loader": batchloader.BatchLoader(
            test_dataset, collate_fn, batch_size=batch_size
        ),
        "cifar100_train": train_dataset.train_image_set,
        "cifar100_test": test_dataset.test_image_set,
    }


def cifar10(batch_size=128):
    train_dataset, test_dataset = get_cifar10()

    def collate_fn(batch):
        return tuple(
            torch.stack(z) if isinstance(z[0], torch.Tensor) else torch.tensor(z)
            for z in zip(*batch)
        )

    return {
        "train_loader": batchloader.BatchLoader(
            train_dataset, collate_fn, batch_size=batch_size, shuffle=True
        ),
        "test_loader": batchloader.BatchLoader(
            test_dataset, collate_fn, batch_size=batch_size
        ),
        "val_loader": batchloader.BatchLoader(
            test_dataset, collate_fn, batch_size=batch_size
        ),
    }
