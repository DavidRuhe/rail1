import math
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
from .pointnet import PointNet
from .cnf import ConditionalNeuralField


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        input_dim,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim_out = dim_out
        self.w0 = w0
        self.c = c
        self.is_first = is_first

        self.weight = nn.Parameter(torch.empty(dim_out, input_dim))
        self.bias = nn.Parameter(torch.empty(dim_out)) if bias else None

        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        dim = self.input_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
        self.weight.data.uniform_(-w_std, w_std)

        if self.bias is not None:
            self.bias.data.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class Modulator(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList([])
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[0] + dims[i], dims[i + 1]))

    def forward(self, x):
        res = x
        for layer in self.layers:
            x = layer(torch.cat([x, res], dim=-1))
            x = torch.relu(x)
        return x


class SirenCNF(nn.Module):
    def __init__(
        self,
        input_dim,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        bias=True,
        final_activation=None,
        dropout=0.0,
        modulator_dims=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.w0 = w0
        self.w0_initial = w0_initial
        self.bias = bias
        self.final_activation = final_activation
        self.dropout = dropout
        assert len(modulator_dims) == num_layers + 1
        self.modulator = Modulator(modulator_dims) if modulator_dims is not None else None

        if self.modulator is not None:
            self.forward = self.forward_conditional

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            is_first = i == 0
            layer_w0 = w0_initial if is_first else w0
            layer_input_dim = input_dim if is_first else dim_hidden

            layer = Siren(
                input_dim=layer_input_dim,
                dim_out=dim_hidden,
                w0=layer_w0,
                bias=bias,
                is_first=is_first,
                dropout=dropout,
            )

            self.layers.append(layer)

        final_activation = (
            nn.Identity() if final_activation is None else final_activation
        )
        self.last_layer = Siren(
            input_dim=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            bias=bias,
            activation=final_activation,
        )

    def forward_conditional(self, x, c):

        c = self.modulator(c)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)
            x *= torch.sigmoid(c[i])

        return self.last_layer(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class SCNF(nn.Module):

    def __init__(self, backbone="pointnet", *args, **kwargs):
        super().__init__()
        assert backbone == "pointnet"
        self.backbone = PointNet()

        self.cnf = SirenCNF(*args, **kwargs)

    def forward(self, x_pos, x_neg, idx):

        z = self.backbone(x_pos, idx)

        result = self.cnf(torch.cat([x_pos, x_neg], dim=0), z.repeat(2, 1))
        return result
