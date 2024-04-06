"""The resampled Modelnet40 dataset by Stanford researchers."""

from datasets.modelnet40_stf import ModelNet40STF, LABEL_TO_IDX, pc_normalize
from sklearn.cluster import KMeans
import torch
from threadpoolctl import threadpool_limits
import fpsample
from torch_geometric.nn import fps
from rail1.data import batchloader
import numpy as np


class Modelnet40KMeans(ModelNet40STF):

    def __init__(self, num_points, deterministic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_points = num_points
        self.n_down = 3
        self.deterministic= deterministic


    def __getitem__(self, index):
        points, label = super().__getitem__(index)
        points = points[:, :3]

        if self.transforms is not None:
            for t in self.transforms:
                points = t(points)

        return points, label

        # points = pc_normalize(points)

        # # Select num_points randomly
        # idx = np.random.choice(len(points), self.num_points, replace=False)
        # points = points[idx]


        # all_points = [points]
        # with threadpool_limits(limits=1, user_api="openmp"):
        #     for i in range(self.n_down):
        #         points = torch.from_numpy(points)

        #         if self.deterministic:
        #             p = torch.cat([points.mean(0, keepdim=True), points])
        #             c = fps(p, ratio=0.5, random_start=False).numpy()
        #             c = p[c[1:]]
        #         else:
        #             c = "random"

        #         kmeans = KMeans(
        #             n_clusters=len(points) // 2,
        #             max_iter=32,
        #             tol=1e-4,
        #             init=c,
        #             n_init=1,
        #         )
        #         result = kmeans.fit(points.numpy())
        #         points = result.cluster_centers_
        #         all_points.append(points)

        return all_points, label


def load_modelnet40stf_points_kmeans(
    num_points, *, batch_size=32, num_workers=4, n_prefetch=2, deterministic=True
):
    train = Modelnet40KMeans(
        num_points=num_points,
        train=True,
        deterministic=deterministic,
    )
    test = Modelnet40KMeans(
        num_points=num_points,
        train=False,
        deterministic=deterministic,
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


if __name__ == "__main__":
    modelnet40 = Modelnet40KMeans(num_points=1024, train=True)
    # all_points, _ = modelnet40[1]

    import matplotlib.pyplot as plt

    # # 3D plot all points
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, nrows=1, ncols=4)
    # for i, points in enumerate(all_points):
    #     ax[i].scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    # plt.savefig("modelnet40_kmeans.png")

    # # Time batchloader
    # import time

    # print(len(modelnet40))

    # loader = batchloader.BatchLoader(
    #     modelnet40, batch_size=2, shuffle=False, num_workers=0, n_prefetch=0
    # )
    # t0 = time.time()
    # # for i in range(1):
    # #     points, label = loader[i]

    # t1 = time.time()
    # print("Time", t1 - t0)

    # points, label = loader[0]

    points = modelnet40[1][0]

    # 3d plot all points
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, nrows=1, ncols=4)

    for i in range(len(points)):
        ex = points[i]
        print(ex.shape)
        ax[i].scatter(ex[:, 0], ex[:, 1], ex[:, 2], s=1)
    plt.savefig("modelnet40_kmeans.png")
    print("Done")
