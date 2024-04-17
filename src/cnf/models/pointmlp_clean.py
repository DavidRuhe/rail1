import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functional import conv_mlp, pctools

from .functional import conv_mlp
from .functional.activation import get_activation
from .pointnet import PointConv


class GeometricAffineModule(nn.Module):

    def __init__(self, channels, mode="center", cat_anchor=True):
        super().__init__()

        self.cat_anchor = cat_anchor
        self.mode = mode.lower()

        if self.mode == "center":
            self.normalize = self._normalize_center
        elif self.mode == "anchor":
            self.normalize = self._normalize_anchor
        else:
            raise ValueError(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor]."
            )
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channels]))

    def _normalize_center(self, grouped_features, pos, features):
        return torch.mean(grouped_features, dim=2, keepdim=True)

    def _normalize_anchor(self, grouped_features, pos, features):

        mean = torch.cat([features, pos], dim=-1) if pos is not None else features
        mean = mean.unsqueeze(dim=-2)
        return mean

    def forward(self, grouped_pos, pos, grouped_features, features):
        """Learned affine renormalization of the grouped features.

        Args:
            grouped_pos: (B, M, K, D) coordinates of the groups in grouped_pos
            pos: (B, M, D) coordinates of the points in pos
            grouped_features: (B, M, K, C) features of the group members
            features: (B, M, C) features of the points

        Returns:
            grouped_features: (B, M, K, C) renormalized features
        """

        # Normalize the coordinates.
        grouped_pos, *_ = pctools.recenter_groups(
            grouped_pos, pos, grouped_features, features
        )

        # Normalize the features.
        mean = self.normalize(grouped_features, pos, features)
        std = torch.std(grouped_features - mean, dim=(1, 2, 3), keepdim=True)
        grouped_features = (grouped_features - mean) / (std + 1e-5)
        grouped_features = self.affine_alpha * grouped_features + self.affine_beta

        return grouped_pos, pos, grouped_features, features


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        k,
        expansion=1.0,
        activation="relu",
        use_xyz=True,
        normalize_mode="center",
    ):
        super().__init__()
        self.conv = PointConv(
            k=k,
            mlp=nn.Sequential(
                conv_mlp.conv2d_mlp(
                    [2 * channels + 3 * use_xyz, out_channels],
                    bn=True,
                    activation=activation,
                ),
                conv_mlp.conv2d_mlp(
                    [out_channels, int(out_channels * expansion), out_channels],
                    bn=True,
                    residual=True,
                    activation=activation,
                ),
            ),
            normalizer=GeometricAffineModule(
                channels + 3 * use_xyz, mode=normalize_mode
            ),
            cat_features=True
        )

    def forward(self, pos_features, idx):
        return self.conv(pos_features, idx)


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        expansion=1.0,
        activation="relu",
    ):
        super(PosExtraction, self).__init__()
        self.mlp = conv_mlp.conv1d_mlp(
            [channels, int(channels * expansion), channels],
            bn=True,
            activation=activation,
            residual=True,
        )

    def forward(self, pos_features):
        pos, features = pos_features
        return pos, self.mlp(features.transpose(1, 2)).transpose(1, 2)


class PointMLP(nn.Module):
    def __init__(
        self,
        points=1024,
        class_num=40,
        embed_dim=64,
        expansion=1.0,
        activation="relu",
        use_xyz=True,
        normalize_mode="center",
        channel_mult=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
    ):
        super().__init__()
        assert (
            len(pre_blocks) == len(k_neighbors) == len(pos_blocks) == len(channel_mult)
        ), "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points

        self.embedding = conv_mlp.conv1d_mlp([3, embed_dim], bn=True, activation=activation)

        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        for i in range(len(pre_blocks)):
            out_channel = last_channel * channel_mult[i]
            kneighbor = k_neighbors[i]

            pre_block_module = PreExtraction(
                last_channel,
                out_channel,
                kneighbor,
                expansion=expansion,
                activation=activation,
                use_xyz=use_xyz,
                normalize_mode=normalize_mode,
            )

            self.pre_blocks_list.append(pre_block_module)
            pos_block_module = PosExtraction(
                out_channel,
                expansion=expansion,
                activation=activation,
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.global_pool = lambda x: torch.max(x, 1).values
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num),
        )

    def forward(self, all_points, idx):
        """PointMLP forward pass.

        Args:
            all_points: torch.Tensor, shape [B, N, 3]
            idx: list of torch.Tensor, shape [B, M_i, K]
        """

        pos = pctools.index(all_points, idx[0])
        features = pos
        features = self.embedding(features.transpose(1, 2)).transpose(1, 2)

        pos_features = (pos, features)

        for i in range(self.stages):

            pos_features = self.pre_blocks_list[i](pos_features, idx[i + 1])  # Conv
            pos_features = self.pos_blocks_list[i](pos_features)  # Update

        pos, features = pos_features
        features = self.global_pool(features)
        return self.classifier(features)


def construct_pointmlp(num_classes=40):
    return PointMLP(
        points=1024,
        class_num=num_classes,
        embed_dim=64,
        expansion=1.0,
        activation="relu",
        use_xyz=False,
        channel_mult=[2, 2, 2],
        pre_blocks=[2, 2, 2],
        pos_blocks=[2, 2, 2],
        k_neighbors=[
            24,
            24,
            24,
        ],
    )
