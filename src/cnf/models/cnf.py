import torch
from torch import nn


class ZeroLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ZeroLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class FiLMLinear(nn.Module):
    def __init__(self, in_features, out_features, c_features=None):
        super(FiLMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c_features = c_features

        if c_features is not None:
            self.gamma = ZeroLinear(c_features, in_features)
            self.beta = ZeroLinear(c_features, in_features)
            self.forward = self.forward_conditional  # type: ignore

        self.linear = nn.Linear(in_features, out_features)

    def forward_conditional(self, x, c):
        x = self.linear(x)
        return x * (1 + self.gamma(c)) + self.beta(c)

    def forward(self, x):
        return self.linear(x)


class ConditionalNeuralField(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(256, 128, 64),
        input_conditioning_dim=None,
        conditioning_hidden_dim=256,
        activation=nn.GELU(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.conditioning_dim = input_conditioning_dim

        if input_conditioning_dim is not None:
            self.forward = self.forward_conditional  # type: ignore
            self.c_emb = nn.Linear(input_conditioning_dim, conditioning_hidden_dim)
        else:
            conditioning_hidden_dim = None

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.network.append(
                FiLMLinear(hidden_dims[i], hidden_dims[i + 1], conditioning_hidden_dim)
            )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward_conditional(self, x, c):
        c = self.c_emb(c)
        x = self.activation(self.input_layer(x))
        for layer in self.network:
            x = self.activation(layer(x, c))
        x = self.output_layer(x)
        return x

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.network:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
