import torch
from torch import nn
from torch.nn import init


# class Bilinear(nn.Module):
#     def __init__(self, dim = 1024):
#         super().__init__()
#         self.linear_1 = nn.Linear(dim, dim)
#         self.linear_2 = nn.Linear(dim, dim)
#         self.a = nn.Parameter(torch.zeros(dim))
#         # self.linear_3 = nn.Linear(dim, dim)


#     def forward(self, x):
#         b = self.linear_2(x) / (torch.sigmoid(self.a[None])*(x.abs() - 1) + 1)
#         return self.linear_1(x) * b + x


class Bilinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(input_dim, output_dim)
        self.a = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        h = self.linear_1(x)
        b = h / (torch.sigmoid(self.a[None]) * (h.abs() - 1) + 1)
        return self.linear_2(x) * b


class RandomPointsAE(nn.Module):

    def __init__(
        self, num_points=1, input_dim=3, dims=(512, 512, 512), activation=nn.GELU()
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dims = dims
        self.activation = activation
        self.num_points = num_points

        layers = []
        layers.append(Bilinear(input_dim, dims[0]))
        for i in range(len(dims) - 1):
            layers.append(Bilinear(dims[i], dims[i + 1]))
        # layers.append(nn.Linear(input_dim, dims[0]))
        # layers.append(activation)
        # for i in range(len(dims) - 1):
        #     layers.append(nn.Linear(dims[i], dims[i + 1]))
        #     layers.append(activation)

        self.layers = nn.Sequential(*layers)

        self.head = nn.Linear(dims[-1], 3 * num_points)
        # Initialize head with zeros
        init.zeros_(self.head.weight)
        init.constant_(self.head.bias, 1 / 3)

    def forward(self, cdist, basis):
        z = self.layers(cdist)
        h = self.head(z).reshape(len(z), self.num_points, 3)
        output = torch.bmm(h, basis)
        return output


class RandomSurfacesMLP(nn.Module):

    def __init__(
        self, num_points=1, input_dim=3, dims=(512, 512, 512), activation=nn.GELU()
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dims = dims
        self.activation = activation
        self.num_points = num_points

        layers = []
        layers.append(Bilinear(input_dim, dims[0]))
        for i in range(len(dims) - 1):
            layers.append(Bilinear(dims[i], dims[i + 1]))
        # layers.append(nn.Linear(input_dim, dims[0]))
        # layers.append(activation)
        # for i in range(len(dims) - 1):
        #     layers.append(nn.Linear(dims[i], dims[i + 1]))
        #     layers.append(activation)

        self.layers = nn.Sequential(*layers)

        self.head = nn.Linear(dims[-1], 3 * num_points)
        # Initialize head with zeros
        init.zeros_(self.head.weight)
        init.constant_(self.head.bias, 1 / 3)

    def forward(self, cdist):
        z = self.layers(cdist)
        return z
