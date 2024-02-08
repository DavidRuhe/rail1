import torch
from torch import nn
import functools


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         attn_pool=None,
#         pre_norm=True,
#         drop_rate=0.0,
#         global_pool="avg",
#         norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
#         head_layer=nn.Linear,
#     ):
#         super().__init__()
#         if attn_pool is not None:
#             raise NotImplementedError

#         self.global_pool = global_pool
#         self.attn_pool = attn_pool
#         self.pre_norm = pre_norm

#         self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
#         self.norm = norm_layer(input_dim) if pre_norm else nn.Identity()
#         self.head = head_layer(input_dim, output_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.attn_pool is not None:
#             x = self.attn_pool(x)
#         elif self.global_pool == "avg":
#             # x = x[:, self.num_prefix_tokens :].mean(dim=1)
#             x = x.mean(dim=1)

#         x = self.drop(x)

#         x = self.norm(x)

#         return self.head(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Decoder
        cls = x.mean(0)
        raise
        x = self.mlp_head(cls)
        
        return x


# if __name__ == "__main__":
#     decoder = Decoder(10, 10)
#     print(decoder)

#     x = torch.randn(1, 32, 10)
#     print(decoder(x))
