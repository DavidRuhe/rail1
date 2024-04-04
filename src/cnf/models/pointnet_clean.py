import torch
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from torch import nn
import math


def pointnet_mlp(mlp_spec, bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


def knn(x, y, k):
    return torch.cdist(y, x).topk(k, dim=2, largest=False)


def index_at(x, idx):
    """Indexes a tensor x [B, N, C] along the first dimension with idx [B, M, K] to get [B, M, K, C]"""

    return torch.gather(
        x.unsqueeze(1).expand(-1, idx.shape[1], -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]),
    )


class PointNetLayer(nn.Module):
    def __init__(self, k, mlp_spec, bn=True, use_xyz=True):
        super().__init__()
        self.k = k
        self.use_xyz = use_xyz

        if use_xyz:
            mlp_spec[0] += 3

        self.mlp = pointnet_mlp(mlp_spec, bn)

    def forward(self, loc, new_loc, features):

        # Get the k nearest points in loc for each point in new_loc
        knn_dist, knn_idx = knn(loc, new_loc, self.k)

        knn_loc = index_at(loc, knn_idx)

        knn_loc = knn_loc.permute(0, 3, 1, 2)

        # Center the points around new_loc
        knn_loc -= new_loc.transpose(1, 2).unsqueeze(-1)

        knn_features = knn_loc
        if features is not None:
            knn_features = index_at(features.transpose(1, 2), knn_idx).permute(
                0, 3, 1, 2
            )
            if self.use_xyz:
                knn_features = torch.cat([knn_loc, knn_features], dim=1)  #

        features = self.mlp(knn_features)
        features = torch.max(features, 3)[0]

        return features


class PointNetFinal(nn.Module):
    def __init__(self, k, mlp_spec, bn=True, use_xyz=True):
        super().__init__()
        self.k = k
        self.use_xyz = use_xyz

        if use_xyz:
            mlp_spec[0] += 3

        self.mlp = pointnet_mlp(mlp_spec, bn)

    def forward(self, loc, features):
        loc = loc.transpose(1, 2).unsqueeze(2)
        if features is not None:
            features = features.unsqueeze(2)
            if self.use_xyz:
                features = torch.cat([loc, features], dim=1)
        else:
            features = loc

        features = self.mlp(features)
        features = torch.max(features, 3)[0]
        return features


class PointNetPPClassification(nn.Module):
    def __init__(self, use_xyz=True, kmeans=False):
        super().__init__()

        self.use_xyz = use_xyz
        self.kmeans = kmeans

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetLayer(
                k=64,
                mlp_spec=[0, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointNetLayer(mlp_spec=[128, 128, 128, 256], use_xyz=self.use_xyz, k=64)
        )
        self.final = PointNetFinal(
            mlp_spec=[256, 256, 512, 1024],
            k=64,
            use_xyz=self.use_xyz,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, pointcloud):
        features = None

        for i, module in enumerate(self.SA_modules):

            features = module(pointcloud[i], pointcloud[i + 1], features)

        features = self.final(pointcloud[i + 1], features)

        return self.fc_layer(features.squeeze(-1))
