from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from models.functional import mlp, pctools


class PointConv(nn.Module):
    def __init__(self, k, mlp_spec, bn=True, use_xyz=True):
        super().__init__()
        self.k = k
        self.use_xyz = use_xyz

        if use_xyz:
            mlp_spec[0] += 3

        self.mlp_spec = mlp_spec

        self.mlp = mlp.conv2d_mlp(mlp_spec, bn)
        self.grouper = partial(
            pctools.group_to_idx,
            query_fn=partial(pctools.knn, k=k),
            normalize_fn=pctools.recenter_groups,
        )

        self.aggr = lambda tensor: torch.max(tensor, -2).values

    def forward(self, pos_features, idx):
        pos, grouped_features = self.grouper(pos_features, idx)
        grouped_features = self.mlp(grouped_features.transpose(1, -1)).transpose(1, -1)
        features = self.aggr(grouped_features)
        return (pos, features)


class PointNetPPClassification(nn.Module):
    def __init__(self, use_xyz=True, kmeans=False):
        super().__init__()

        self.use_xyz = use_xyz
        self.kmeans = kmeans

        self._build_model()

    def _build_model(self):
        self.convnet = nn.ModuleList()
        self.convnet.append(
            PointConv(
                k=64,
                mlp_spec=[3, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.convnet.append(
            PointConv(mlp_spec=[128, 128, 128, 256], use_xyz=self.use_xyz, k=64)
        )

        self.global_mlp = mlp.conv1d_mlp(mlp_spec=[256 + 3, 256, 512, 1024])
        self.global_pool = lambda x: torch.max(x, -1).values

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

    def forward(self, all_points, idx):
        """PointNet forward pass.

        Args:
            all_points: torch.Tensor, shape [B, N, 3]
            idx: list of torch.Tensor, shape [B, M_i, K]
        """

        pos = pctools.index(all_points, idx[0])
        features = pos
        pos_features = (pos, features)

        for i, module in enumerate(self.convnet):
            pos_features = module(pos_features, idx[i + 1])

        pos, features = pos_features

        features = torch.cat([features, pos], dim=-1)
        features = self.global_mlp(features.transpose(1, 2))
        features = self.global_pool(features)

        return self.fc_layer(features.squeeze(-1))
