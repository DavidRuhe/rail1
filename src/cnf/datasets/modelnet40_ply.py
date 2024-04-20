import glob
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from .modelnet40_stf import LABEL_TO_IDX

DATAROOT = os.environ["DATAROOT"]


def load_data(partition):
    all_data = []
    all_label = []

    for h5_name in glob.glob(
        os.path.join(
            DATAROOT, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition
        )
    ):
        f = h5py.File(h5_name)
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, stages=[1024, 512, 256, 128], partition="train"):
        self.data, self.label = load_data(partition)
        self.partition = partition
        self.stages = stages

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == "train":
            pointcloud = random_point_dropout(
                pointcloud
            )  # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        idx = []
        for i, stage in enumerate(self.stages):
            if i == 0:
                high = pointcloud.shape[0]
            else:
                high = self.stages[i - 1]
            stage_idx = np.random.randint(0, high, (stage,))
            idx.append(stage_idx)


        return (pointcloud, idx), label

    def __len__(self):
        return self.data.shape[0]


def load_modelnet40_ply(*, batch_size=32, num_workers=4, n_prefetch=2):

    train = ModelNet40(partition="train")
    test = ModelNet40(partition="test")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}

    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": test_loader,
        "idx_to_label": idx_to_label,
    }
