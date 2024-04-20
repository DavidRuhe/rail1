from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import conv_mlp, pctools, mlp
from .pointnet import PointConv


class TransitionDown(nn.Module):
    def __init__(self, k, channels):
        super().__init__()

        self.conv = PointConv(
            k=k,
            mlp=conv_mlp.conv2d_mlp(channels),
            cat_pos=True,
        )

    def forward(self, pos, points):
        idx = torch.randint(0, pos.size(1), (pos.size(0), pos.size(1) // 2))
        new_pos, new_points = self.conv((pos, points), idx)
        return new_pos, new_points


class TransformerBlock(nn.Module):
    def __init__(self, channels_in, channels_out, k) -> None:
        super().__init__()
        self.embedding = nn.Linear(channels_in, channels_out)
        self.linear_out = nn.Linear(channels_out, channels_in)

        self.mlp_posenc = mlp.mlp(
            [3, channels_out, channels_out], act_last=False, bn=False
        )
        self.mlp_attn = mlp.mlp(
            [channels_out, channels_out, channels_out], act_last=False, bn=False
        )
        self.w_qs = nn.Linear(channels_out, channels_out, bias=False)
        self.w_ks = nn.Linear(channels_out, channels_out, bias=False)
        self.w_vs = nn.Linear(channels_out, channels_out, bias=False)
        self.k = k

        normalizer = pctools.recenter_groups
        self.grouper = partial(
            pctools.group_to_idx,
            query_fn=partial(pctools.knn, k=k),
            normalize_fn=normalizer,
        )

    def forward(self, pos, features):

        # Group using all poins.
        idx = torch.arange(pos.size(1), device=pos.device)[None].expand(len(pos), -1)
        grouped_pos, pos, _, _, knn_idx = self.grouper((pos, features), idx)

        residual = features

        x = self.embedding(features)
        # Vector attention (eq. 2 of the paper)
        q, k, v = (
            self.w_qs(x),  # [B, N, C]
            pctools.index(self.w_ks(x), knn_idx),  # [B, N, K, C]
            pctools.index(self.w_vs(x), knn_idx),  # [B, N, K, C]
        )

        posenc = self.mlp_posenc(grouped_pos)

        attn = self.mlp_attn(q[:, :, None] - k + posenc)  # beta := subtraction
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        result = torch.einsum("bmnf,bmnf->bmf", attn, v + posenc)
        result = self.linear_out(result) + residual

        return result


class Backbone(nn.Module):
    def __init__(self, transformer_dim, k, nblocks):
        super().__init__()
        self.embedding = mlp.mlp([3, 32, 32], bn=False)
        self.transformer1 = TransformerBlock(32, transformer_dim, k)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    k,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(TransformerBlock(channel, transformer_dim, k))
        self.nblocks = nblocks

    def forward(self, x):
        pos = x
        features = self.transformer1(pos, self.embedding(x))
        for i in range(self.nblocks):
            pos, features = self.transition_downs[i](pos, features)
            features = self.transformers[i](pos, features)
        return features


class PointTransformerCls(nn.Module):
    def __init__(self):
        super().__init__()
        nblocks = 4
        nneighbor = 16
        n_c = 40
        transformer_dim = 512
        self.backbone = Backbone(
            transformer_dim=transformer_dim,
            k=nneighbor,
            nblocks=nblocks,
        )
        self.classifier = mlp.mlp(
            [transformer_dim, 256, 128, n_c], bn=True, act_last=False, dropout=0.5
        )

        self.nblocks = nblocks
        self.global_pool = lambda x: x.max(1).values

    def forward(self, x, idx):
        x = pctools.index(x, idx[0])
        features = self.backbone(x)
        features = self.global_pool(features)
        res = self.classifier(features)
        return res
