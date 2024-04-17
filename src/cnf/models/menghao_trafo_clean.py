import torch
import torch.nn as nn

# from pointnet2_ops.pointnet2_utils import farthest_point_sample, index_points, square_distance
# from .pointnet import knn, index
from models.functional import mlp, pctools
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


class SelfAttentionLayer(nn.Module):
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
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        # self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # self.bn1 = nn.BatchNorm1d(channels)
        # self.bn2 = nn.BatchNorm1d(channels)
        # self.l1 = conv1d_batchnorm_act(channels, channels)
        # self.l2 = conv1d_batchnorm_act(channels, channels)

        self.sa1 = SelfAttentionLayer(channels)
        self.sa2 = SelfAttentionLayer(channels)
        self.sa3 = SelfAttentionLayer(channels)
        self.sa4 = SelfAttentionLayer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        # x = self.relu(self.bn2(self.conv2(x)))
        x = self.l1(x)
        x = self.l2(x)

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

        self.embedding = mlp.conv1d_mlp([d_points, 64, 64])
        # self.gather_local_0 = PreExtraction(in_channels=128, out_channels=128)
        # self.gather_local_1 = PreExtraction(in_channels=256, out_channels=256)
        self.conv1 = PointConv(
            k=32,
            mlp=mlp.conv2d_mlp([128, 128, 128], bn=True),
            cat_features=True,
        )
        self.conv2 = PointConv(
            k=32,
            mlp=mlp.conv2d_mlp([128, 128, 256], bn=True),
            cat_features=True,
        )

        self.trafo = StackedAttention()

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        self.relu = nn.ReLU()

    def forward(self, all_points, idx):
        """PointMLP forward pass.

        Args:
            all_points: torch.Tensor, shape [B, N, 3]
            idx: list of torch.Tensor, shape [B, M_i, K]
        """

        pos = pctools.index(all_points, idx[0])
        features = pos
        features = self.embedding(features.transpose(1, 2)).transpose(1, 2)

        # features = features.transpose(1, 2)
        # new_xyz = index(xyz, indices[1])
        # new_features = index(features, indices[1])
        # # x = x.permute(0, 2, 1)
        # # new_xyz, new_features = sample_and_group(
        # #     npoint=512, nsample=32, xyz=xyz, points=x
        # # )
        # new_features = knn_and_center(
        #     xyz, new_xyz, features, new_features, 32, concat_xyz=True
        # )
        # features = self.gather_local_0(new_features)
        # # feature = feature_0.permute(0, 2, 1)
        # # new_xyz, new_features = sample_and_group(
        # #     npoint=256, nsample=32, xyz=new_xyz, points=feature
        # # )
        # xyz = new_xyz
        # new_xyz = index(xyz, indices[2])
        # new_features = index(features, indices[2])
        # new_features = knn_and_center(
        #     xyz, new_xyz, features, new_features, 32, concat_xyz=True
        # )
        # features = self.gather_local_1(new_features)

        pos_features = (pos, features)
        breakpoint()
        pos_features = self.conv1(pos_features, idx[1])
        pos_features = self.conv2(pos_features, idx[2])

        breakpoint()

        features = features.transpose(1, 2)
        x = self.trafo(features)

        x = torch.cat([x, features], dim=1)

        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x
