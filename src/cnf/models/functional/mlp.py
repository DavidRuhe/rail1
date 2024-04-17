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
    elif activation.lower() == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unknown activation function: {activation}.")


def convnd_mlp(mlp_spec, convnd, bn=None, residual=False, activation="relu"):
    """
    Creates a nD convolutional MLP.

    The network will accept a [B, C, D1, D2, ..., Dn] tensor
    and apply a series of 1x1 convolutions followed by
    batch normalization and ReLU activations. Output
    shape will be [B, C', D1, D2, ..., Dn].

    Args:
        mlp_spec (List[int]): list of integers specifying the
            number of channels in each layer
        convnd (nn.Module): the nD convolutional layer
        bn (bool): whether to use batch normalization
        residual (bool): whether to use residual connections
        activation (str): the activation function to use

    Returns:
        nn.Sequential: the network
    """
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(convnd(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn))
        if bn is not None:
            layers.append(bn(mlp_spec[i]))
        layers.append(get_activation(activation))

    seq = nn.Sequential(*layers)
    if residual:
        seq = Residual(seq)
    return seq


def conv2d_mlp(mlp_spec, bn: bool = True, residual=False, activation="relu"):
    """
    Creates a 2D convolutional MLP.

    The network will accept a [B, C, D1, D2] tensor
    and apply a series of 1x1 convolutions followed by
    batch normalization and ReLU activations. Output
    shape will be [B, C', D1, D2].

    Args:
        mlp_spec (List[int]): list of integers specifying the
            number of channels in each layer
        bn (bool): whether to use batch normalization
        residual (bool): whether to use residual connections
        activation (str): activation function to use

    Returns:
        nn.Sequential: the network
    """
    return convnd_mlp(
        mlp_spec, nn.Conv2d, nn.BatchNorm2d if bn else None, residual, activation
    )


def conv1d_mlp(mlp_spec, bn: bool = True, residual=False, activation="relu"):
    """
    Creates a 1D convolutional MLP.

    The network will accept a [B, C, D1] tensor
    and apply a series of 1x1 convolutions followed by
    batch normalization and ReLU activations. Output
    shape will be [B, C', D1].

    Args:
        mlp_spec (List[int]): list of integers specifying the
            number of channels in each layer
        bn (bool): whether to use batch normalization
        residual (bool): whether to use residual connections
        activation (str): activation function to use

    Returns:
        nn.Sequential: the network
    """
    return convnd_mlp(
        mlp_spec, nn.Conv1d, nn.BatchNorm1d if bn else None, residual, activation
    )
