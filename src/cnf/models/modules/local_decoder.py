import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_coordinate(p, padding=0.1, plane="xz"):
    """Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    """
    if plane == "xz":
        xy = p[:, :, [0, 2]]
    elif plane == "xy":
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    """Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class LocalDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        d_dim=None,
    ):
        super().__init__()
        # c_dim = 32
        if d_dim:
            self.c_dim = d_dim
        else:
            self.c_dim = c_dim

        self.n_blocks = n_blocks

        if self.c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(self.c_dim, hidden_size) for i in range(n_blocks)]
            )

        self.fc_p = nn.Linear(3, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.th = nn.Tanh()

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode
        ).squeeze(-1)

        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, **kwargs):

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        out = self.th(out)
        return out
