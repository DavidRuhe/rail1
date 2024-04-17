import torch
import torch.nn as nn

# from pointnet2_ops.pointnet2_utils import farthest_point_sample, index_points, square_distance
# from .pointnet import knn, index
from models.functional import conv_mlp, pctools
from .pointnet import PointConv


# def sample_and_group(npoint, nsample, xyz, points):
#     B, N, C = xyz.shape
#     S = npoint

#     fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

#     new_xyz = index_points(xyz, fps_idx)
#     new_points = index_points(points, fps_idx)

#     dists = square_distance(new_xyz, xyz)  # B x npoint x N
#     idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

#     grouped_points = index_points(points, idx)
#     grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
#     new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
#     return new_xyz, new_points



def knn_and_center(
    loc, new_loc, features, new_features, k, concat_xyz=True, concat_new_features=True
):
    knn_dist, knn_idx = knn(loc, new_loc, k)

    knn_loc = index(loc, knn_idx)

    # Center the points around new_loc
    knn_loc -= new_loc[:, :, None]

    knn_features = knn_loc
    if features is not None:
        knn_features = index(features, knn_idx)
        if concat_new_features:
            knn_features = torch.cat(
                [knn_features, new_features[:, :, None].expand(-1, -1, k, -1)], dim=-1
            )
            # if concat_xyz:
        #     knn_features = torch.cat([knn_loc, knn_features], dim=-1)
    return knn_features


class PreExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU()
        self.l1 = conv1d_batchnorm_act(in_channels, out_channels)
        self.l2 = conv1d_batchnorm_act(out_channels, out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        # x = x.permute(0, 1, 3, 2)
        # x = x.reshape(-1, d, s)
        x = x.transpose(2, 3).reshape(b * n, d, s)
        x = self.l1(x)
        x = self.l2(x)
        x = x.max(dim=2).values
        x = x.reshape(b, n, -1)  # [B, N, D]
        return x


class ConvSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, C, N]
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy / (x_q.size(-1) ** 0.5))
        x_r = x_v @ attention  # b, c, n
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
        # x: [B, C, N]
        x = self.mlp(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerClsClean(nn.Module):
    def __init__(self):
        super().__init__()
        d_points = 3
        output_channels = 40

        self.embedding = conv_mlp.conv1d_mlp([d_points, 64, 64])
        # self.gather_local_0 = PreExtraction(in_channels=128, out_channels=128)
        # self.gather_local_1 = PreExtraction(in_channels=256, out_channels=256)
        self.conv1 = PointConv(
            k=32,
            mlp=conv_mlp.conv2d_mlp([128, 128, 128], bn=True),
            cat_features=True,
        )
        self.conv2 = PointConv(
            k=32,
            mlp=conv_mlp.conv2d_mlp([256, 256, 256], bn=True),
            cat_features=True,
        )

        self.trafo = StackedAttention(channels=256)
        self.conv_fuse = conv_mlp.conv1d_mlp([256 + 1024, 1024], bn=True)
        self.global_pool = lambda x: torch.max(x, 1).values

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_channels),
        )

        self.global_pool = lambda x: torch.max(x, 1).values

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

        pos_features = self.conv1(pos_features, idx[1])
        pos_features = self.conv2(pos_features, idx[2])

        pos, features = pos_features
        
        h = self.trafo(features.transpose(1, 2)).transpose(1, 2)

        x = torch.cat([h, features], dim=2)

        x = self.conv_fuse(x.transpose(1, 2)).transpose(1, 2)
        x = self.global_pool(x)

        x = self.classifier(x)

        return x
