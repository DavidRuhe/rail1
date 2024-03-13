import os
from rail1.data import batchloader
from rail1.data import collate

from .embedded_mnist import load_embedded_mnist
from .shapenet import load_shapenet
from .shapenet_s2vs import ShapeNet, AxisScaling
from .toy_points import load_random_points_dataset


def random_points(
    n_points=1, dim=3, batch_size=128, num_workers=0, num_prefetch=0, return_basis=False,
):
    return load_random_points_dataset(
        n_points=n_points,
        dim=dim,
        batch_size=batch_size,
        num_workers=num_workers,
        num_prefetch=num_prefetch,
        return_basis=return_basis,
    )


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
        **kwargs,
    )
    val_dataset = load_shapenet(
        mode="val",
        split="val",
        dataset_folder=dataset_folder,
        return_idx=True,
        **kwargs,
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


def shapenet_s2vs(batch_size, **kwargs):

    transform = AxisScaling((0.75, 1.25), True)
    train_set = ShapeNet(
        split="train",
        transform=transform,
        sampling=True,
        num_samples=1024,
        return_surface=True,
        surface_sampling=True,
        **kwargs,
    )
    val_set = ShapeNet(
        split="val",
        transform=None,
        sampling=False,
        return_surface=True,
        surface_sampling=True,
        **kwargs,
    )
    test_set = ShapeNet(
        split="test",
        transform=None,
        sampling=False,
        return_surface=True,
        surface_sampling=True,
        **kwargs,
    )

    train_loader = batchloader.BatchLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        n_prefetch=2,
        shuffle=True,
    )

    val_loader = batchloader.BatchLoader(
        val_set,
        batch_size=1,
        num_workers=4,
        n_prefetch=2,
        shuffle=False,
    )

    test_loader = batchloader.BatchLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        n_prefetch=2,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader,
        "train_dataset": train_set,
        "test_dataset": test_set,
    }
