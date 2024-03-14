import os
import kaolin
from kaolin.io import modelnet
import numpy as np
from tqdm import tqdm

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


class ModelNet40:
    def __init__(self, root, split, num_points=None):
        self.root = root
        self.split = split

        self.memmap_points = np.lib.format.open_memmap(
            os.path.join(self.root, f"{self.split}_points.npy"),
            mode="c",
            dtype=np.float32,
        )
        self.memmap_labels = np.lib.format.open_memmap(
            os.path.join(self.root, f"{self.split}_labels.npy"),
            mode="c",
            dtype=np.int64,
        )

        self.num_points = num_points

    def __len__(self):
        return len(self.memmap_labels)

    def __getitem__(self, idx):
        points = self.memmap_points[idx]

        if self.num_points is not None:
            idxs = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[idxs]

        label = self.memmap_labels[idx]
        return points, label


def load_modelnet40_points(batch_size=32, num_points=1024):
    dataroot = os.environ["DATAROOT"]
    train = ModelNet40(
        num_points=num_points, root=os.path.join(dataroot, "ModelNet40_points"), split="train"
    )
    test = ModelNet40(
        num_points=num_points, root=os.path.join(dataroot, "ModelNet40_points"), split="test"
    )
    train_loader = batchloader.BatchLoader(
        train,
        batch_size=batch_size,
        num_workers=4,
        n_prefetch=2,
        shuffle=True,
    )

    test_loader = batchloader.BatchLoader(
        test,
        batch_size=batch_size,
        num_workers=4,
        n_prefetch=2,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": test_loader,
    }


def preprocess_modelnet40(num_points=2048):

    dataroot = os.environ["DATAROOT"]
    output_path = os.path.join(dataroot, f"ModelNet40_points")
    os.makedirs(output_path, exist_ok=True)

    modelnet40_train = modelnet.ModelNet(
        os.path.join(dataroot, "ModelNet40"), split="train", output_dict=True
    )
    modelnet40_test = modelnet.ModelNet(
        os.path.join(dataroot, "ModelNet40"), split="test", output_dict=True
    )

    print("Processing train set...")
    memmap_train_points = np.lib.format.open_memmap(
        os.path.join(output_path, "train_points.npy"),
        mode="w+",
        shape=(len(modelnet40_train), num_points, 3),
        dtype=np.float32,
    )
    memmap_train_labels = np.lib.format.open_memmap(
        os.path.join(output_path, "train_labels.npy"),
        mode="w+",
        shape=(len(modelnet40_train),),
        dtype=np.int64,
    )

    for idx, mesh in enumerate(tqdm(modelnet40_train)):
        vertices = mesh["mesh"].vertices.unsqueeze(dim=0).float()
        faces = mesh["mesh"].faces.long()
        try:
            points, _ = kaolin.ops.mesh.sample_points(vertices, faces, num_points)
            points = points.squeeze(dim=0)
        except ValueError:
            print(f"Could not sample mesh {idx} with label {mesh['label']}. Skipping.")
            continue
        points = points.squeeze(dim=0)
        memmap_train_points[idx] = points.numpy()
        memmap_train_labels[idx] = LABEL_TO_IDX[mesh["label"]]

    del memmap_train_points
    del memmap_train_labels

    print("Processing test set...")
    memmap_test_points = np.lib.format.open_memmap(
        os.path.join(output_path, "test_points.npy"),
        mode="w+",
        shape=(len(modelnet40_test), num_points, 3),
        dtype=np.float32,
    )
    memmap_test_labels = np.lib.format.open_memmap(
        os.path.join(output_path, "test_labels.npy"),
        mode="w+",
        shape=(len(modelnet40_test),),
        dtype=np.int64,
    )

    for idx, mesh in enumerate(tqdm(modelnet40_test)):
        vertices = mesh["mesh"].vertices.unsqueeze(dim=0).float()
        faces = mesh["mesh"].faces.long()
        try:
            points, _ = kaolin.ops.mesh.sample_points(vertices, faces, num_points)
            points = points.squeeze(dim=0)
        except ValueError:
            print(f"Could not sample mesh {idx} with label {mesh['label']}. Skipping.")
            continue
        memmap_test_points[idx] = points.numpy()
        memmap_test_labels[idx] = LABEL_TO_IDX[mesh["label"]]
    del memmap_test_points
    del memmap_test_labels

    print("Done.")


if __name__ == "__main__":
    preprocess_modelnet40()
