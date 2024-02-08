"""Inspired by Huggingface's TIMM"""
import functools

import torch
from rail1.utils import python
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        dropout=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features or in_features)

        bias = python.to_2tuple(bias)
        drop_probs = python.to_2tuple(dropout)
        linear_layer = (
            functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )

        self.norm1 = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])  # type: ignore
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0]) if drop_probs[0] > 0.0 else nn.Identity()  # type: ignore

        self.norm2 = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])  # type: ignore
        self.drop2 = nn.Dropout(drop_probs[1]) if drop_probs[1] > 0.0 else nn.Identity()  # type: ignore

    def forward(self, x):
        x = self.drop1(x)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)

        x = self.drop2(x)
        x = self.norm2(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        q_norm=False,
        kv_norm=False,
        dropout: float = 0.0,
        add_bias_kv: bool = False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.q_norm = norm_layer(dim) if q_norm else nn.Identity()
        self.kv_norm = norm_layer(dim) if kv_norm else nn.Identity()

        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, add_bias_kv=add_bias_kv, batch_first=True
        )

    def forward(self, h, c=None):
        if c is None:
            c = h
        else:
            c = torch.cat((h, c), dim=1)

        q = self.q_norm(h)
        k = self.kv_norm(c)
        v = self.kv_norm(c)

        return self.attn(q, k, v)[0]


class ProcessorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        add_bias_kv: bool = False,
        q_norm: bool = True,
        kv_norm: bool = True,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=lambda h: nn.LayerNorm(h, eps=1e-6),
        mlp=MLP,
    ):
        super().__init__()

        if drop_path > 0.0:
            raise NotImplementedError
        del drop_path

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            add_bias_kv=add_bias_kv,
            q_norm=q_norm,
            kv_norm=kv_norm,
            dropout=attn_drop,
            norm_layer=norm_layer,
        )
        self.mlp = mlp(dim, dim * mlp_ratio, dim, dropout=mlp_drop, act_layer=act_layer)

    def forward(self, x):
        h, c = x
        h = h + self.attn(h, c)
        h = h + self.mlp(h)
        return (h, c)


class Processor(nn.Module):
    def __init__(
        self,
        num_cores,
        core_dim,
        num_heads,
        mlp_ratio=4.0,
        add_bias_kv=False,
        q_norm=True,
        kv_norm=True,
        mlp_dropout=0.0,
        attn_dropout=0.0,
        core_dropout=0.0,
        input_dropout=0.0,
        input_norm=False,
        depth=12,
        norm_layer=lambda h: nn.LayerNorm(h, eps=1e-6),
    ):
        super().__init__()
        self.core_dim = core_dim

        self.processor = nn.Sequential(
            *[
                ProcessorBlock(
                    core_dim,
                    num_heads,
                    mlp_drop=mlp_dropout,
                    attn_drop=attn_dropout,
                    mlp_ratio=mlp_ratio,
                    q_norm=q_norm,
                    kv_norm=kv_norm,
                    add_bias_kv=add_bias_kv,
                )
                for _ in range(depth)
            ]
        )

        self.input_dropout = (
            nn.Dropout(input_dropout) if input_dropout > 0.0 else nn.Identity()
        )
        self.core_dropout = (
            nn.Dropout(core_dropout) if core_dropout > 0.0 else nn.Identity()
        )

        self.input_norm = norm_layer(core_dim) if input_norm else nn.Identity()

        self.h = nn.parameter.Parameter(torch.zeros(1, num_cores, core_dim))

    def forward(self, c):
        h = self.h.repeat(len(c), 1, 1)

        c = self.input_dropout(c)
        h = self.core_dropout(h)

        c = self.input_norm(c)

        x = self.processor((h, c))
        h, c = x
        return h
