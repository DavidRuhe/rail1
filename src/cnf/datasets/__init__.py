import os
from rail1.data import batchloader
from rail1.data import collate

from .embedded_mnist import load_embedded_mnist
from .shapenet import load_shapenet


def embedded_mnist(batch_size=128):
    return load_embedded_mnist(
        "runs/notebooks/9utazgqo/latent_space/", subset=None, batch_size=batch_size
    )


def shapenet(batch_size=128, **kwargs):
    dataset_folder = os.path.join(os.environ["DATAROOT"], "ShapeNet")
    train_dataset = load_shapenet(
        mode="train",
        split="train",
        dataset_folder=dataset_folder,
        return_idx=False,
        **kwargs
    )
    val_dataset = load_shapenet(
        mode="val",
        split="val",
        dataset_folder=dataset_folder,
        return_idx=True,
        **kwargs
    )

    test_dataset = val_dataset

    train_loader = batchloader.BatchLoader(
        train_dataset,
        collate_fn=collate.default_collate,
        batch_size=batch_size,
        num_workers=4,
        n_prefetch=2,
        shuffle=True,
    )
    val_loader = batchloader.BatchLoader(
        val_dataset,
        collate_fn=collate.default_collate,
        batch_size=1,
        num_workers=4,
        n_prefetch=2,
        shuffle=False,
    )

    test_loader = val_loader

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }
