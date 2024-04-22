import torch
from torch import nn


class Bilinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_1 = nn.Linear(input_dim, output_dim)
        # self.linear_1 = ZeroLinear(input_dim, output_dim)
        self.linear_2 = nn.Linear(input_dim, output_dim)
        # self.a = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        h = self.linear_1(x2)
        # b = h / (torch.sigmoid(self.a[None]) * (h.abs() - 1) + 1)
        # return self.linear_2(x) * b
        return torch.sigmoid(h) * self.linear_2(x1)



class ZeroLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ZeroLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class FiLMLinear(nn.Module):
    def __init__(self, in_features, out_features, c_features=None, custom_factor=1):
        super(FiLMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c_features = c_features

        if c_features is not None:
        #     self.gamma = ZeroLinear(c_features, in_features)
        #     self.beta = ZeroLinear(c_features, in_features)
            self.forward = self.forward_conditional  # type: ignore

        # self.linear = Bilinear(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        # self.custom_factor = custom_factor
        # self.cond_linear = nn.Linear(c_features, in_features)
        self.cond_linear = nn.Linear(c_features, out_features)
        # self.cond_linear = ZeroLinear(c_features, out_features)
        # self.linear = 

    def forward_conditional(self, x, c):
        # x: [B, N, C]
        # c: [B, C]
        # res = x
        # # a = self.custom_factor * (self.gamma(c[:, None]))
        # # b = self.custom_factor * self.beta(c[:, None])
        # x = torch.sigmoid(self.cond_linear(c))
        # return self.linear(x[:, None] * res)
        h = self.cond_linear(c)
        return torch.sigmoid(h[:, None]) * self.linear(x)
        # return self.linear(x, c)

    def forward(self, x):
        return self.linear(x)


class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class ConditionalNeuralField(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(512, 512, 512, 512, 512),
        input_conditioning_dim=None,
        conditioning_hidden_dim=512,
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.conditioning_dim = input_conditioning_dim

        if input_conditioning_dim is not None:
            self.forward = self.forward_conditional  # type: ignore
            # self.c_emb = nn.Linear(input_conditioning_dim, conditioning_hidden_dim)
            self.c_emb = nn.Sequential(
                nn.Linear(input_conditioning_dim, conditioning_hidden_dim),
                activation,
                nn.Linear(conditioning_hidden_dim, conditioning_hidden_dim),
                activation,
            )
        else:
            conditioning_hidden_dim = None

        self.input_layer = nn.Linear(input_dim + conditioning_hidden_dim, hidden_dims[0])
        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.network.append(
                FiLMLinear(hidden_dims[i], hidden_dims[i + 1], conditioning_hidden_dim)
            )
        # self.output_layer = Bilinear(hidden_dims[-1], output_dim)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.modulator = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.modulator.append(
                nn.Linear(2 * conditioning_hidden_dim, conditioning_hidden_dim)
            )


    def forward_conditional(self, x, c):
        c = self.c_emb(c)
        c0 = c
        input = torch.cat([c[:, None].expand(-1, x.size(1), -1), x], dim=-1)
        x = self.activation(self.input_layer(input))
        for i, layer in enumerate(self.network):
            x = self.activation(layer(x, c))
            c = self.activation(self.modulator[i](torch.cat([c0, c], dim=-1)))
        x = self.output_layer(x)
        return x

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.network:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
