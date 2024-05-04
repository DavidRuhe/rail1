import os
from rail1.data import batchloader
from rail1.data import collate

from .embedded_mnist import load_embedded_mnist
from .shapenet import load_shapenet
from .shapenet_s2vs import ShapeNet, AxisScaling
from .random_points import load_random_points_dataset

# from .random_surfaces import load_random_surface_dataset
from .modelnet40_points import load_modelnet40_points
from .modelnet40_stf import load_modelnet40stf_points
from .modelnet40_kmeans import load_modelnet40stf_points_kmeans
from .modelnet40_ply import load_modelnet40_ply
from .random_spheres import load_random_spheres_dataset
from .random_surfaces import load_random_shapes_dataset
from .bunny import load_bunny_dataset

def bunny(*args, **kwargs):
    return load_bunny_dataset(*args, **kwargs)

def random_shapes(*args, **kwargs):
    return load_random_shapes_dataset(*args, **kwargs)


def random_spheres(*args, **kwargs):
    return load_random_spheres_dataset(*args, **kwargs)


def modelnet40_ply(**kwargs):
    return load_modelnet40_ply(**kwargs)


def modelnet40_points(**kwargs):
    return load_modelnet40_points(**kwargs)


def modelnet40stf_points(**kwargs):
    return load_modelnet40stf_points(**kwargs)


def modelnet40stf_points_kmeans(**kwargs):
    return load_modelnet40stf_points_kmeans(**kwargs)


def random_points(
    num_points=1,
    dim=3,
    batch_size=128,
    num_workers=0,
    n_prefetch=0,
    return_basis=False,
):
    return load_random_points_dataset(
        num_points=num_points,
        dim=dim,
        batch_size=batch_size,
        num_workers=num_workers,
        num_prefetch=n_prefetch,
        return_basis=return_basis,
    )


def random_surfaces(
    num_points_per_shape=2048,
    **kwargs,
):
    return load_random_surface_dataset(
        num_points_per_shape=num_points_per_shape,
        **kwargs,
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
