from torch import nn


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
