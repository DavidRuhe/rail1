from functools import partial

import torch
from torch import nn

from models.functional import mlp, pctools


class PointConv(nn.Module):
    def __init__(
        self,
        k,
        mlp: nn.Module,
        normalizer=pctools.recenter_groups,
        use_xyz=False,
        cat_features=False,
    ):
        super().__init__()
        self.k = k
        self.mlp = mlp
        self.use_xyz = use_xyz
        self.cat_features = cat_features
        self.normalizer = normalizer
        self.grouper = partial(
            pctools.group_to_idx,
            query_fn=partial(pctools.knn, k=k),
            normalize_fn=self.normalizer,
        )

        self.aggr = lambda tensor: torch.max(tensor, -2).values

    def forward(self, pos_features, idx):
        grouped_pos, pos, grouped_features, features = self.grouper(pos_features, idx)
        if self.use_xyz:
            grouped_features = torch.cat([grouped_pos, grouped_features], dim=-1)

        if self.cat_features:
            grouped_features = torch.cat(
                [
                    grouped_features,
                    features[:, :, None].expand(-1, -1, grouped_features.size(2), -1),
                ],
                dim=-1,
            )

        grouped_features = self.mlp(grouped_features.transpose(1, -1)).transpose(1, -1)
        features = self.aggr(grouped_features)
        return (pos, features)


class PointNet(nn.Module):
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
                mlp=mlp.conv2d_mlp([3 + 3 * self.use_xyz, 64, 64, 128], bn=True),
            )
        )
        self.convnet.append(
            PointConv(
                mlp=mlp.conv2d_mlp([128 + 3 * self.use_xyz, 128, 128, 256], bn=True),
                k=64,
            )
        )

        self.global_mlp = mlp.conv1d_mlp(mlp_spec=[256 + 3, 256, 512, 1024])
        self.global_pool = lambda x: torch.max(x, 1).values

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
        features = self.global_mlp(features.transpose(1, 2)).transpose(1, 2)
        features = self.global_pool(features)

        return self.fc_layer(features.squeeze(-1))
