import torch
from torch import nn
import torch.nn.functional as F
from models.points_ae import Bilinear


class FPSNet(nn.Module):

    def __init__(
        self,
        #  input_channels, output_dim, k
    ):
        super().__init__()

        # self.input_channels = input_channels
        # self.output_dim = output_dim
        # self.k = k

        # self.weight_mlp = nn.Linear(input_channels, output_dim)
        n_points = 2
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    Bilinear(3, 128),
                    Bilinear(128, 128),
                    Bilinear(128, 128)
                )
                for _ in range(n_points)
            ]
        )

        self.final = nn.Linear(128, 2)

    def forward(self, input):

        
        results = []

        fps = input
        perm = torch.randperm(fps.size(1))
        fps_perm = fps[:, perm]
        norm_expanded = fps.norm(dim=-1, keepdim=True).expand(-1, -1, fps.size(1))
        norm_perm_expanded = fps_perm.norm(dim=-1, keepdim=True).expand(-1, -1, fps.size(1))

        h = torch.stack(
            [
                # torch.einsum("bni, bmi->bnm", input, input),
                torch.cdist(fps, fps),
                norm_expanded,
                norm_expanded.transpose(1, 2),
            ],
            dim=-1,
        )
        h_perm = torch.stack(
            [
                # torch.einsum("bni, bmi->bnm", input, input),
                torch.cdist(fps_perm, fps_perm),
                norm_perm_expanded,
                norm_perm_expanded.transpose(1, 2),
            ],
            dim=-1,
        )

        h = self.mlps[0](h)
        h_perm = self.mlps[0](h_perm)
        attention = F.softmax(h, dim=-2)
        attention_perm = F.softmax(h_perm, dim=-2)
        fps = torch.einsum("bmnc, bni->bmic", attention, fps)
        fps_perm = torch.einsum("bmnc, bni->bmic", attention_perm, fps_perm)
        fps = fps.sum(1).transpose(1, 2)
        fps_perm = fps_perm.sum(1).transpose(1, 2)
        fps = torch.einsum('bnd, on->bod', fps, self.final.weight)
        fps_perm = torch.einsum('bnd, on->bod', fps_perm, self.final.weight)

        # breakpoint()
        results.append(fps[:, :2])

        return torch.cat([
            input[:, :1],
            *results
        ], dim=1)


        # w = self.weight_mlp(input).transpose(1, 2)
        # c = (w @ input) / input.size(1) ** 0.5

        # B, N, _ = input.size()
        # B, C, _ = c.size()

        # h = torch.cat([
        #     torch.einsum("bci,bni->bcn", c, input),
        #     torch.square(c).sum(dim=-1)[:, None].expand((-1, N, -1)),
        #     torch.square(input).sum(dim=-1, keepdim=True).expand((-1, N, -1))
        # ])

        # _, idx = torch.topk(h, self.k, dim=-1, largest=True, sorted=False)
        # idx = idx.view(B, -1)

        # return idx
