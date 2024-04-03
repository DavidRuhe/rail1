import torch
from torch import nn
import torch.nn.functional as F
from models.points_ae import Bilinear


class TransposeNet(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            Bilinear(8, 128), Bilinear(128, 128), Bilinear(128, 128)
        )

        self.x_lin = nn.Linear(2, 2)

        self.final = nn.Linear(128, 2)
        self.normnet = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 4))
        self.w = nn.Parameter(torch.rand(2, 2))

    # def forward(self, x):

    #     x = x.transpose(1, 2)  # [B, C, N, D]
    #     perm = torch.randperm(x.size(2))
    #     x_perm = x[:, :, perm]
    #     norm_expanded = x.norm(dim=-1, keepdim=True).expand(-1, -1, -1, x.size(2))
    #     norm_perm_expanded = x.norm(dim=-1, keepdim=True).expand(-1, -1, -1, x.size(2))
    #     h = torch.stack(
    #         [
    #             torch.einsum("bcnd, bcmd->bcnm", x, x),
    #             torch.cdist(x, x),
    #             norm_expanded,
    #             norm_expanded.transpose(2, 3),
    #         ],
    #         dim=-1,
    #     )
    #     h_perm = torch.stack(
    #         [
    #             torch.einsum("bcnd, bcmd->bcnm", x_perm, x_perm),
    #             torch.cdist(x_perm, x_perm),
    #             norm_perm_expanded,
    #             norm_perm_expanded.transpose(2, 3),
    #         ],
    #         dim=-1,
    #     )

    #     h = h.permute(0, 2, 3, 1, 4).reshape(h.shape[0], *h.shape[2:4], -1)
    #     h_perm = h_perm.permute(0, 2, 3, 1, 4).reshape(h_perm.shape[0], *h_perm.shape[2:4], -1)
    #     h = self.mlp(h)
    #     h_perm = self.mlp(h_perm)

    #     attention = F.softmax(h, dim=-2)
    #     attention_perm = F.softmax(h_perm, dim=-2)

    #     x = torch.einsum('bind, oi->bond', x, self.x_lin.weight)
    #     x_perm = torch.einsum('bind, oi->bond', x_perm, self.x_lin.weight)

    #     c = torch.einsum("bnmf, bond->bomdf", attention, x)
    #     c_perm = torch.einsum("bnmf, bond->bomdf", attention_perm, x_perm)

    #     c = c.sum(2)
    #     c_perm = c_perm.sum(2)

    #     h = torch.einsum("bcdi, oi->bcod", c, self.final.weight)
    #     h_perm = torch.einsum("bcdi, oi->bcod", c_perm, self.final.weight)
    #     return h

    def forward(self, x):

        x_perm = torch.stack([x[:, 1], x[:, 0]], dim=1)

        norm = x.norm(dim=-1, keepdim=True)
        norm_perm = x_perm.norm(dim=-1, keepdim=True)

        w = self.normnet(norm)
        w_perm = self.normnet(norm_perm)

        w = w.reshape(*w.shape[:2], 2, 2)
        w_perm = w_perm.reshape(*w_perm.shape[:2], 2, 2)

        c = torch.einsum("bnmo, bncd->bmod", w, x)
        c_perm = torch.einsum("bnmo, bncd->bmod", w_perm, x_perm)

        # # c = torch.einsum('bncm, bncd->bcmd', w, x)
        # # c_perm = torch.einsum('bncm, bncd->bcmd', w_perm, x_perm)

        # # c1 = x[:, 0].norm(dim=-1, keepdim=True) * x[:, 0] - x[:, 1].norm(dim=-1, keepdim=True) * x[:, 1]
        # # c2 = -x[:, 0].norm(dim=-1, keepdim=True) * x[:, 0] + x[:, 1].norm(dim=-1, keepdim=True) * x[:, 1]

        f00 = lambda x: self.w[0, 0] * x.norm(dim=-1, keepdim=True)
        f01 = lambda x: self.w[0, 1] * x.norm(dim=-1, keepdim=True)
        f10 = lambda x: self.w[1, 0] * x.norm(dim=-1, keepdim=True)
        f11 = lambda x: self.w[1, 1] * x.norm(dim=-1, keepdim=True)

        # # c1 = self.w[0, 0] * x[:, 0].norm(dim=-1, keepdim=True) * x[:, 0] + self.w[0, 1] * x[:, 1].norm(dim=-1, keepdim=True) * x[:, 1]
        # # c2 = self.w[1, 0] * x[:, 0].norm(dim=-1, keepdim=True) * x[:, 0] + self.w[1, 1] * x[:, 1].norm(dim=-1, keepdim=True) * x[:, 1]

        # c1 = f00(x[:, 0]) * x[:, 0] + f01(x[:, 1]) * x[:, 1]
        # c2 = f10(x[:, 0]) * x[:, 0] + f11(x[:, 1]) * x[:, 1]

        # # c1_perm = x_perm[:, 0].norm(dim=-1, keepdim=True) * x_perm[:, 0] - x_perm[:, 1].norm(dim=-1, keepdim=True) * x_perm[:, 1]
        # # c2_perm = -x_perm[:, 0].norm(dim=-1, keepdim=True) * x_perm[:, 0] + x_perm[:, 1].norm(dim=-1, keepdim=True) * x_perm[:, 1]

        # # c1_perm = self.w[0, 0] * x_perm[:, 0].norm(dim=-1, keepdim=True) * x_perm[:, 0] + self.w[0, 1] * x_perm[:, 1].norm(dim=-1, keepdim=True) * x_perm[:, 1]
        # # c2_perm = self.w[1, 0] * x_perm[:, 0].norm(dim=-1, keepdim=True) * x_perm[:, 0] + self.w[1, 1] * x_perm[:, 1].norm(dim=-1, keepdim=True) * x_perm[:, 1]

        # c1_perm = f00(x_perm[:, 0]) * x_perm[:, 0] + f01(x_perm[:, 1]) * x_perm[:, 1]
        # c2_perm = f10(x_perm[:, 0]) * x_perm[:, 0] + f11(x_perm[:, 1]) * x_perm[:, 1]

        # c = torch.stack([c1, c2], dim=2)
        # c_perm = torch.stack([c1_perm, c2_perm], dim=2)

        z = torch.einsum("bmod,bncd->bmno", c, x)
        z_perm = torch.einsum("bmod,bncd->bmno", c_perm, x_perm)

        w = torch.softmax(z, dim=2)
        w_perm = torch.softmax(z_perm, dim=2)

        res = torch.einsum("bncd, bmno->bmod", x, w)
        res_perm = torch.einsum("bncd, bmno->bmod", x_perm, w_perm)

        return res
