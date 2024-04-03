"""The resampled Modelnet40 dataset by Stanford researchers."""

import os

import numpy as np
import torch.utils.data as data
import tqdm
from rail1.data import batchloader

LABEL_TO_IDX = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 30,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39,
}


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40STF(data.Dataset):
    def __init__(
        self, num_points, deterministic=True, transforms=None, train=True, download=True
    ):
        super().__init__()

        self.num_points = num_points
        self.transforms = transforms
        self.train = train
        self.deterministic = deterministic

        if train:
            data_path = os.path.join(
                os.environ["DATAROOT"], "modelnet40_stanford", "train.npy"
            )
            label_path = os.path.join(
                os.environ["DATAROOT"], "modelnet40_stanford", "train_labels.npy"
            )
        else:
            data_path = os.path.join(
                os.environ["DATAROOT"], "modelnet40_stanford", "test.npy"
            )
            label_path = os.path.join(
                os.environ["DATAROOT"], "modelnet40_stanford", "test_labels.npy"
            )

        self.memmap_data = np.lib.format.open_memmap(
            data_path,
            mode="c",
            dtype=np.float32,
        )
        self.memmap_labels = np.lib.format.open_memmap(
            label_path,
            mode="c",
            dtype=np.int64,
        )

        self.length = len(self.memmap_labels)
        self.indices = np.arange(10_000)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        points = self.memmap_data[idx]
        label = self.memmap_labels[idx]

        if self.deterministic:
            idx = np.arange(self.num_points)
        else:
            idx = np.random.choice(self.indices, self.num_points, replace=False)
        points = points[idx][:, :3]

        max_norm = np.max(np.linalg.norm(points, axis=1))
        points = (points - np.mean(points, axis=0)) / max_norm

        if self.transforms is not None:
            raise NotImplementedError
        return points, label


def load_modelnet40stf_points(
    num_points, *, batch_size=32, num_workers=4, n_prefetch=2
):
    train = ModelNet40STF(
        num_points=num_points,
        train=True,
    )
    test = ModelNet40STF(
        num_points=num_points,
        train=False,
    )

    train_loader = batchloader.BatchLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=n_prefetch,
        shuffle=True,
    )

    test_loader = batchloader.BatchLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        n_prefetch=n_prefetch,
        shuffle=False,
    )

    idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}

    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": test_loader,
        "idx_to_label": idx_to_label,
    }


def preprocess_modelnet40():

    root = os.path.join(os.environ["DATAROOT"], "modelnet40_normal_resampled")
    assert os.path.exists(root)
    newroot = os.path.join(os.environ["DATAROOT"], "modelnet40_stanford")
    if not os.path.exists(newroot):
        os.makedirs(newroot)

    train_objects = os.path.join(root, "modelnet40_train.txt")
    test_objects = os.path.join(root, "modelnet40_test.txt")

    train_objects = open(train_objects).read().split("\n")[:-1]
    test_objects = open(test_objects).read().split("\n")[:-1]

    num_train = len(train_objects)
    num_test = len(test_objects)

    train_data_mmap = np.lib.format.open_memmap(
        os.path.join(newroot, "train.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_train, 10_000, 6),
    )

    test_data_mmap = np.lib.format.open_memmap(
        os.path.join(newroot, "test.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(num_test, 10_000, 6),
    )

    train_labels = np.lib.format.open_memmap(
        os.path.join(newroot, "train_labels.npy"),
        mode="w+",
        dtype=np.int64,
        shape=(num_train,),
    )

    test_labels = np.lib.format.open_memmap(
        os.path.join(newroot, "test_labels.npy"),
        mode="w+",
        dtype=np.int64,
        shape=(num_test,),
    )

    for i, obj in enumerate(tqdm.tqdm(train_objects)):
        cls = obj.rsplit("_", 1)[0]
        path = os.path.join(root, cls, obj) + ".txt"
        points = np.loadtxt(path, delimiter=",").astype(np.float32)
        train_data_mmap[i] = points
        train_labels[i] = LABEL_TO_IDX[cls]

    for i, obj in enumerate(tqdm.tqdm(test_objects)):
        cls = obj.rsplit("_", 1)[0]
        path = os.path.join(root, cls, obj) + ".txt"
        points = np.loadtxt(path, delimiter=",").astype(np.float32)
        test_data_mmap[i] = points
        test_labels[i] = LABEL_TO_IDX[cls]

    del train_data_mmap
    del test_data_mmap
    del train_labels
    del test_labels


if __name__ == "__main__":
    preprocess_modelnet40()
