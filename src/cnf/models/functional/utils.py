import torch
from torch import nn


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]


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
