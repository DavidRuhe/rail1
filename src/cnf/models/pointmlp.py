import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_clean import knn, index
from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "rrelu":
        return nn.RReLU(inplace=True)
    elif activation.lower() == "selu":
        return nn.SELU(inplace=True)
    elif activation.lower() == "silu":
        return nn.SiLU(inplace=True)
    elif activation.lower() == "hardswish":
        return nn.Hardswish(inplace=True)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:

        return nn.ReLU(inplace=True)


def conv1d_batchnorm_act(
    in_channels, out_channels, kernel_size=1, bias=True, activation="relu"
):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias),
        nn.BatchNorm1d(out_channels),
        get_activation(activation),
    )


import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, sub_module):
        super(Residual, self).__init__()
        self.sub_module = sub_module

    def forward(self, x):
        return self.sub_module(x) + x


def conv1d_batchnorm_act_res(
    channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation="relu"
):
    act = (
        nn.ReLU() if activation == "relu" else nn.LeakyReLU()
    )  # assuming 'leakyrelu' for non-ReLU activations
    net1 = nn.Sequential(
        nn.Conv1d(
            in_channels=channel,
            out_channels=int(channel * res_expansion),
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
        ),
        nn.BatchNorm1d(int(channel * res_expansion)),
        act,
    )
    if groups > 1:
        net2 = nn.Sequential(
            nn.Conv1d(
                in_channels=int(channel * res_expansion),
                out_channels=channel,
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm1d(channel),
            act,
            nn.Conv1d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(channel),
        )
    else:
        net2 = nn.Sequential(
            nn.Conv1d(
                in_channels=int(channel * res_expansion),
                out_channels=channel,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(channel),
        )

    return Residual(nn.Sequential(net1, net2))


# Example usage:
# residual_block = conv1d_batchnorm_act_res(64, kernel_size=3, groups=1, res_expansion=0.5, bias=True, activation='relu')
# output = residual_block(input_tensor)


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
        use_xyz=True,
    ):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = conv1d_batchnorm_act(
            in_channels, out_channels, bias=bias, activation=activation
        )
        operation = []
        for _ in range(blocks):
            operation.append(
                conv1d_batchnorm_act_res(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
    ):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                # ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
                conv1d_batchnorm_act(
                    channels, channels, bias=bias, activation=activation
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class LocalGrouper(nn.Module):
    def __init__(
        self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs
    ):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor]."
            )
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(
                torch.ones([1, 1, 1, channel + add_channel])
            )
            self.affine_beta = nn.Parameter(
                torch.zeros([1, 1, 1, channel + add_channel])
            )

    def forward(self, xyz, features, new_xyz, new_features):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        # new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        # new_features = index_points(points, fps_idx)  # [B, npoint, d]

        # idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        # grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        # grouped_features = index_points(points, idx)  # [B, npoint, k, d]

        # knn_dist, knn_idx = knn(xyz, new_xyz, self.kneighbors)

        knn_dist, knn_idx = knn(xyz, new_xyz, self.kneighbors)
        grouped_xyz = index(xyz, knn_idx)  # [B, npoint, k, 3]
        grouped_features = index(features, knn_idx)  # [B, npoint, k, d]

        if self.use_xyz:
            grouped_features = torch.cat(
                [grouped_features, grouped_xyz], dim=-1
            )  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_features, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = (
                    torch.cat([new_features, new_xyz], dim=-1)
                    if self.use_xyz
                    else new_features
                )
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]

            # std = torch.std((grouped_features-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            std = torch.std(grouped_features - mean, dim=(1, 2, 3), keepdim=True)
            grouped_features = (grouped_features - mean) / (std + 1e-5)
            grouped_features = self.affine_alpha * grouped_features + self.affine_beta

        new_features = torch.cat(
            [
                grouped_features,
                new_features[:, :, None].expand(-1, -1, self.kneighbors, -1),
            ],
            dim=-1,
        )
        return new_features


class PointMLP(nn.Module):
    def __init__(
        self,
        points=1024,
        class_num=40,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=True,
        normalize="center",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        **kwargs,
    ):
        super().__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = conv1d_batchnorm_act(
            3, embed_dim, bias=bias, activation=activation
        )
        assert (
            len(pre_blocks)
            == len(k_neighbors)
            == len(reducers)
            == len(pos_blocks)
            == len(dim_expansion)
        ), "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(
                last_channel, anchor_points, kneighbor, use_xyz, normalize
            )  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(
                last_channel,
                out_channel,
                pre_block_num,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
                use_xyz=use_xyz,
            )
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(
                out_channel,
                pos_block_num,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

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

    def forward(self, points, indices):

        loc = index(points, indices[0])
        features = loc.clone()
        batch_size, _, _ = features.size()
        features = self.embedding(features.transpose(1, 2))

        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]

            features = features.transpose(1, 2)
            new_loc = index(loc, indices[i + 1])
            new_features = index(features, indices[i + 1])
            new_features = self.local_grouper_list[i](
                loc, features, new_loc, new_features
            )  # [b,g,3]  [b,g,k,d]

            new_features = self.pre_blocks_list[i](new_features)  # [b,d,g]
            new_features = self.pos_blocks_list[i](new_features)  # [b,d,g]

            loc = new_loc
            features = new_features

        x = F.adaptive_max_pool1d(features, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def pointMLP(num_classes=40, **kwargs):
    return PointMLP(
        points=1024,
        class_num=num_classes,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        # dim_expansion=[2, 2, 2, 2],
        # pre_blocks=[2, 2, 2, 2],
        # pos_blocks=[2, 2, 2, 2],
        # k_neighbors=[24, 24, 24, 24],
        # reducers=[2, 2, 2, 2],
        dim_expansion=[2, 2, 2],
        pre_blocks=[2, 2, 2],
        pos_blocks=[2, 2, 2],
        k_neighbors=[24, 24, 24,],
        reducers=[2, 2, 2],

        **kwargs,
    )