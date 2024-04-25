import torch
from torch import nn
from .pointnet import PointNet
from .cnf import ConditionalNeuralField


class RandomSpheresCNF(nn.Module):

    def __init__(self, backbone="pointnet"):
        super().__init__()
        assert backbone == "pointnet"
        self.backbone = PointNet(channels_out=128)
        self.cnf = ConditionalNeuralField(
            input_dim=3,
            output_dim=1,
            input_conditioning_dim=128,
        )

    def forward(self, queries, points, idx):

        z = self.backbone(points, idx)

        result = self.cnf(queries, z)
        return result

