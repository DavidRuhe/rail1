from torch import nn


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Residual(nn.Module):
    def __init__(self, sub_module):
        super(Residual, self).__init__()
        self.sub_module = sub_module

    def forward(self, x):
        return self.sub_module(x) + x


class BatchNormChannelsLast(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        """x: [B, N1, N2, ..., Nn, C]"""
        B, *N, C = x.size()
        return (
            self.bn(x.view(B, -1, C).transpose(1, 2)).transpose(1, 2).view(B, *N, C)
        )
