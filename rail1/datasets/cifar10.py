import torch
import os
from torchvision import datasets, transforms

from rail1.data import batchloader

DATAROOT = os.environ["DATAROOT"]

# class CustomDataLoader:
#     def __init__(self, dataset, sampler, batch_size=1):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.sampler = sampler
        
#     def __iter__(self):

#         sampler_iter = iter(self.sampler)

#         for batch_idx in range(len(self)):
#             batch_indices = [next(sampler_iter) for _ in range(self.batch_size)]
#             images_labels = [self.dataset[i] for i in batch_indices]
#             images, labels = zip(*images_labels)
#             images = torch.stack(images) / 255
#             labels = torch.tensor(labels)
#             yield images, labels

#     def __len__(self):
#         return len(self.sampler)


def load_cifar10(batch_size=128):
    cifar10_train = datasets.CIFAR10(
        root=DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    # cifar10_test = datasets.CIFAR10(
    #     root=DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    # )

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images) / 255
        labels = torch.tensor(labels)
        return images, labels

    train_loader = batchloader.BatchLoader(
        cifar10_train, collate_fn=collate_fn, batch_size=32, num_workers=4, n_prefetch=4, shuffle=True
    )
    test_loader = batchloader.BatchLoader(
        cifar10_train, collate_fn=collate_fn, batch_size=32, num_workers=4, n_prefetch=4, shuffle=True
    )





    # train_loader = data.DataLoader(
    #     cifar10_train, batch_size=batch_size, sampler=InfiniteRandomSampler(cifar10_train)
    # )
    # test_loader = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    # train_loader = CustomDataLoader(
    #     cifar10_train, data.InfiniteRandomSampler(cifar10_train), batch_size=batch_size
    # )
    # test_loader = CustomDataLoader(
    #     cifar10_test, data.SequentialSampler(cifar10_test), batch_size=batch_size
    # )
    # trai
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": test_loader,
    }
