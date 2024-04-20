from torch import nn

from ..modules.utils import BatchNormChannelsLast, Residual
from .utils import get_activation


def mlp(
    mlp_spec,
    bn: bool = True,
    residual=False,
    activation="relu",
    dropout=0.0,
    act_last=True,
):
    """
    Create a multi-layer perceptron (MLP) neural network. Note: this can be less
    efficient than using a "convolutional" MLP.

    Args:
        mlp_spec (List[int]): List of integers specifying the number of units in each layer.
        bn (bool, optional): Whether to apply batch normalization. Defaults to False.
        residual (bool, optional): Whether to apply residual connections. Defaults to False.
        activation (str, optional): Activation function to use. Defaults to "relu".
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        act_last (bool, optional): Whether to apply activation function to the last layer. Defaults to True.

    Returns:
        nn.Sequential: The MLP neural network model.
    """

    layers = []
    for i in range(1, len(mlp_spec) - 1):
        layers.append(nn.Linear(mlp_spec[i - 1], mlp_spec[i], bias=not bn))

        if bn:
            layers.append(nn.BatchNorm1d(mlp_spec[i]))

        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(mlp_spec[-2], mlp_spec[-1], bias=not (act_last and bn)))
    if act_last:
        if bn:
            layers.append(nn.BatchNorm1d(mlp_spec[-1]))
        layers.append(get_activation(activation))

    seq = nn.Sequential(*layers)
    if residual:
        seq = Residual(seq)
    return seq
