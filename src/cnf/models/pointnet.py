import torch
import torch.nn.functional as F
from functools import partial

# from pointnet2_ops import pointnet2_utils
from torch import nn
import math
from models.functional import pctools, mlp


# def pointnet_mlp(mlp_spec, bn: bool = True):
#     layers = []
#     for i in range(1, len(mlp_spec)):
#         layers.append(
#             nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
#         )
#         if bn:
#             layers.append(nn.BatchNorm2d(mlp_spec[i]))
#         layers.append(nn.ReLU(True))

#     return nn.Sequential(*layers)


# def knn(x, y, k):
#     return torch.cdist(y, x).topk(k, dim=2, largest=False)


# def index(tensor, idx):
#     """
#     Args:
#         tensor: (B, N, C)
#         idx: (B, ..., N) index tensor
#     Returns:
#         new_points: (B, ..., C) indexed points
#     """
#     B, N, D = tensor.shape
#     view_shape = (B,) + (1,) * (len(idx.shape) - 1)
#     repeat_shape = (1,) + idx.shape[1:]
#     batch_indices = (
#         torch.arange(B, dtype=torch.long, device=tensor.device)
#         .view(view_shape)
#         .repeat(repeat_shape)
#     )
#     new_points = tensor[batch_indices, idx, :]
#     return new_points


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


class PointNetFinal(nn.Module):
    def __init__(self, k, mlp_spec, bn=True, use_xyz=True):
        super().__init__()
        self.k = k
        self.use_xyz = use_xyz

        if use_xyz:
            mlp_spec[0] += 3

        self.mlp = mlp.conv2d_mlp(mlp_spec, bn)

    def forward(self, pos_features):
        loc, features = pos_features
        if self.use_xyz:
            features = torch.cat([loc, features], dim=-1)
        else:
            features = loc
        features = self.mlp(features[:, :, None].transpose(1, -1)).transpose(1, -1).squeeze(2)
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
            PointConv(
                k=64,
                mlp_spec=[3, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointConv(mlp_spec=[128, 128, 128, 256], use_xyz=self.use_xyz, k=64)
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

        self.global_pool = lambda x: torch.max(x, 1).values

    def forward(self, all_points, idx):
        """PointNet forward pass.

        Args:
            all_points: torch.Tensor, shape [B, N, 3]
            idx: list of torch.Tensor, shape [B, M_i, K]
        """

        pos = pctools.index(all_points, idx[0])
        features = pos
        pos_features = (pos, features)

        for i, module in enumerate(self.SA_modules):
            pos_features = module(pos_features, idx[i + 1])

        features = self.final(pos_features)
        features = self.global_pool(features)

        return self.fc_layer(features.squeeze(-1))
