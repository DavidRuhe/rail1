from torch import nn

from ..modules.utils import BatchNormChannelsLast, Residual
from .utils import get_activation


def mlp(mlp_spec, bn: bool = False, residual=False, activation="relu"):
    """
    Create a multi-layer perceptron (MLP) neural network. Note: this can be less
    efficient than using a "convolutional" MLP.

    Args:
        mlp_spec (List[int]): List of integers specifying the number of units in each layer.
        bn (bool, optional): Whether to apply batch normalization. Defaults to False.
        residual (bool, optional): Whether to apply residual connections. Defaults to False.
        activation (str, optional): Activation function to use. Defaults to "relu".

    Returns:
        nn.Sequential: The MLP neural network model.
    """

    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(nn.Linear(mlp_spec[i - 1], mlp_spec[i], bias=not bn))
        if bn:
            layers.append(BatchNormChannelsLast(mlp_spec[i]))
        layers.append(get_activation(activation))

    seq = nn.Sequential(*layers)
    if residual:
        seq = Residual(seq)
    return seq
