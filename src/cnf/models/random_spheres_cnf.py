import torch
from torch import nn
from .pointnet import PointNet
from .cnf import ConditionalNeuralField


class RandomSpheresCNF(nn.Module):

    def __init__(self, backbone="pointnet"):
        super().__init__()
        assert backbone == "pointnet"
        self.backbone = PointNet()
        self.cnf = ConditionalNeuralField(
            input_dim=3,
            output_dim=1,
            input_conditioning_dim=40,
        )

    def forward(self, x_pos, x_neg, idx):

        z = self.backbone(x_pos, idx)

        result = self.cnf(torch.cat([x_pos, x_neg], dim=0), z.repeat(2, 1))
        return result

