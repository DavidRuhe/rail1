import torch
from torch import nn
from .modules.fourier_layer import FourierSeriesEmbedding, SigmoidEmbedding
from .pointnet import PointNet
from .siren import SirenNet


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


class SpheresFNF(nn.Module):
    def __init__(self, input_dim, output_dim, input_conditioning_dim):
        super().__init__()

        # self.mlp = nn.Sequential(
        #     Bilinear(2, 256),
        #     Bilinear(256, 256),
        #     Bilinear(256, 256),
        #     Bilinear(256, output_dim)
        # )

        # self.bilinear = Bilinear(4, 4)

        self.mlp = nn.Sequential(
            nn.Linear(512 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

        # self.ff = FourierSeriesEmbedding(5, 256)
        # self.ff = SigmoidEmbedding(5, 256)

    def forward(self, x, r, label):

        n = torch.norm(x, dim=-1, keepdim=True)
        r = r[:, None].expand(-1, x.size(1), -1)
        s = torch.cat([n, r], dim=-1)

        s = self.ff(s).reshape(*x.shape[:-1], -1)

        x = torch.cat([x, s], dim=-1)


        return self.mlp(x)




class ShapesFNF(nn.Module):

    def __init__(self, input_dim, output_dim, input_conditioning_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(32 * 32 + 3 + 5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, output_dim),
        )

        self.ff = FourierSeriesEmbedding(32, 32)
        # self.ff = SirenNet(1, 256, 32, 2)

        self.mlp2 = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.layernorm = nn.LayerNorm(32 * 32)




    def forward(self, x, r, label):

        n = torch.norm(x, dim=-1, keepdim=True)
        r = r[:, None].expand(-1, x.size(1), -1)
        s = torch.cat([n, r], dim=-1)

        s = self.mlp2(s)

        s = self.ff(s).reshape(*x.shape[:-1], -1)
        
        label = label[:, None].expand(-1, x.size(1), -1)

        x = torch.cat([x, s, label], dim=-1)

        return self.mlp(x)


class ConditionalPointNetFNF(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, input_conditioning_dim=None):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(3 + 64 + 1),
            nn.Linear(68, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

        self.ff = FourierSeriesEmbedding(4, 32)
        self.pointnet = PointNet(64, in_features=128)
        # self.ff = SirenNet(33, 256, 256, 2)


    def forward(self, queries, pc, idx):
        
        features = torch.cat([pc, pc.norm(dim=-1, keepdim=True)], dim=-1)
        
        ff = self.ff(features)

        z = self.pointnet(pc, idx, features=ff)

        z = z[:, None].expand(-1, queries.size(1), -1)

        n = torch.norm(queries, dim=-1, keepdim=True)



        x = torch.cat([queries, z, n], dim=-1)

        return self.mlp(x)
    
import math

class RFF(nn.Module):
    def __init__(self, input_dim, hidden_dim, learnable_coefficients=True, std=1.0):
        super(RFF, self).__init__()
        assert hidden_dim % 2 == 0, "hidden_dim should be even."
        self.hidden_dim = hidden_dim
        self.learnable_coefficients = learnable_coefficients
        self.std = std

        self.learnable_coefficients = False
        self.pi = math.pi


        # Embedding layer
        self.coefficients = nn.Linear(input_dim, self.hidden_dim, bias=False)
        # nn.init.zeros_(self.coefficients.bias)
        nn.init.normal_(self.coefficients.weight, mean=0.0, std=1)
        # nn.init.uniform_(self.coefficients.weight, -2 * self.pi * self.hidden_dim, 2 * self.pi * self.hidden_dim)
        # self.postnorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Scaling input by pi
        x = self.pi * x * self.hidden_dim // 2

        # x = x * self.hidden_dim
        
        if self.learnable_coefficients:
            x_proj = self.coefficients(x)
        else:
            # Stop gradient for non-learnable coefficients
            with torch.no_grad():
                x_proj = self.coefficients(x)

        # Apply standard deviation scaling
        # x_proj = x_proj / self.hidden_dim

        # Calculate sin and cos projections
        sin_part = torch.sin(x_proj[..., :self.hidden_dim // 2])
        cos_part = torch.cos(x_proj[..., self.hidden_dim // 2:])
        # sin_part = torch.sin(x_proj)
        # return sin_part

        # Concatenate sin and cos parts
        x =  torch.cat((sin_part, cos_part), dim=-1)

        # x = x / (2 * math.pi * self.hidden_dim)

        # x = self.postnorm(x)
        return x
    

class FNF(nn.Module):
    def __init__(self):
        super().__init__()

        # self.ff = FourierSeriesEmbedding(3, 256)
        self.ff = RFF(1, 64, learnable_coefficients=False)
        self.mlp = nn.Sequential(
            # nn.LayerNorm(192),
            nn.Linear(192, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        ff = self.ff(input[..., None])
        ff = ff.reshape(*input.shape[:-1], -1)
        return self.mlp(ff)


def conditional_pointnet_fnf():
    return ConditionalPointNetFNF()


def shapes_fnf(input_dim, output_dim, input_conditioning_dim):

    return ShapesFNF(input_dim, output_dim, input_conditioning_dim)

def spheres_fnf(input_dim, output_dim, input_conditioning_dim):

    return SpheresFNF(input_dim, output_dim, input_conditioning_dim)

def bunny_fnf(*args, **kwargs):
    return FNF(*args, **kwargs)