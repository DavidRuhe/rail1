import torch
import torch.nn as nn
from .pointmlp import knn, index
import torch.nn.functional as F


def sample_and_group(npoint, nsample, xyz, points, idx, returnfps=False):
    new_xyz = index(xyz, idx)
    knn_dist, knn_idx = knn(xyz, new_xyz, nsample)
    grouped_xyz = index(xyz, knn_idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

    if points is not None:
        grouped_points = index(points, knn_idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, idx):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.nsample, xyz, points, idx)
        new_xyz = index(xyz, idx)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        knn_dist, knn_idx = knn(xyz, xyz, self.k)
        knn_xyz = index(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = (
            self.w_qs(x),
            index(self.w_ks(x), knn_idx),
            index(self.w_vs(x), knn_idx),
        )

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / (k.size(-1) ** 0.5), dim=-2)  # b x n x k x f

        res = torch.einsum("bmnf,bmnf->bmf", attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(
            k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True
        )

    def forward(self, xyz, points, idx):
        return self.sa(xyz, points, idx)


class Backbone(nn.Module):
    def __init__(self, transformer_dim, nneighbor, nblocks, npoints):
        super().__init__()
        self.npoints = npoints
        self.fc1 = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    npoints // 4 ** (i + 1),
                    nneighbor,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(
                TransformerBlock(channel, transformer_dim, nneighbor)
            )
        self.nblocks = nblocks

    def forward(self, x, idx):
        x = index(x, idx[0])
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            
            xyz, points = self.transition_downs[i](xyz, points, idx[i + 1])
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self):
        super().__init__()

        nblocks = 3
        nneighbor = 16
        transformer_dim = 512
        num_point = 1024
        n_c = 40
        self.backbone = Backbone(transformer_dim, nneighbor, nblocks, num_point)
        # npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2**nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c),
        )
        self.nblocks = nblocks

    def forward(self, x, idx):
        points, _ = self.backbone(x, idx)
        res = self.fc2(points.mean(1))
        return res
