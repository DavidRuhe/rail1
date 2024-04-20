import torch
import torch.nn as nn

from .functional import conv_mlp, mlp, pctools
from .pointnet import PointConv


class ConvSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()

        self.mlp = conv_mlp.conv1d_mlp([channels, channels, channels])
        self.sa1 = ConvSelfAttention(channels)
        self.sa2 = ConvSelfAttention(channels)
        self.sa3 = ConvSelfAttention(channels)
        self.sa4 = ConvSelfAttention(channels)

    def forward(self, x):

        x = self.mlp(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointCloudTransformer(nn.Module):
    def __init__(self, output_channels=40):
        super().__init__()

        self.embedding = conv_mlp.conv1d_mlp([3, 64, 64])
        self.conv1 = PointConv(
            k=32,
            mlp=conv_mlp.conv2d_mlp([128, 128, 128]),
            cat_features=True,
        )
        self.conv2 = PointConv(
            k=32,
            mlp=conv_mlp.conv2d_mlp([256, 256, 256]),
            cat_features=True,
        )

        self.trafo = StackedAttention(channels=256)
        self.conv_fuse = conv_mlp.conv1d_mlp([256 + 1024, 1024], bn=True)
        self.classifier = mlp.mlp(
            [1024, 512, 256, output_channels], bn=True, dropout=0.5, act_last=False
        )

        self.global_pool = lambda x: torch.max(x, 2).values

    def forward(self, all_points, idx):
        """PointTransformer forward pass.

        Args:
            all_points: torch.Tensor, shape [B, N, 3]
            idx: list of torch.Tensor, shape [B, M_i, K]
        """

        pos = pctools.index(all_points, idx[0])
        features = pos
        features = self.embedding(features.transpose(1, 2)).transpose(1, 2)

        pos_features = (pos, features)
        pos_features = self.conv1(pos_features, idx[1])
        pos_features = self.conv2(pos_features, idx[2])

        pos, features = pos_features

        features = features.transpose(1, 2)  # [B, N, C] -> [B, C, N]
        h = self.trafo(features)

        x = torch.cat([h, features], dim=1)
        x = self.conv_fuse(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x
