import torch
from rail1.data import batchloader
from datasets.reverse import get_reverse


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
        "val_loader": batchloader.BatchLoader(val_dataset, collate_fn, batch_size=batch_size),
        "test_loader": batchloader.BatchLoader(test_dataset, collate_fn, batch_size=batch_size),
    }
