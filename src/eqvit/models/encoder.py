import math
from enum import Enum
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from rail1 import logging
from rail1.utils import python
from torch import nn

# class Format(str, Enum):
#     NCHW = "NCHW"
#     NHWC = "NHWC"
#     NCL = "NCL"
#     NLC = "NLC"


# def resample_abs_pos_embed(
#     posemb,
#     new_size: List[int],
#     old_size: Optional[List[int]] = None,
#     num_prefix_tokens: int = 1,
#     interpolation: str = "bicubic",
#     antialias: bool = True,
#     verbose: bool = False,
# ):
#     # sort out sizes, assume square if old size not provided
#     num_pos_tokens = posemb.shape[1]
#     num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
#     if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
#         return posemb

#     if old_size is None:
#         hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
#         old_size = hw, hw  # type: ignore

#     if num_prefix_tokens:
#         posemb_prefix, posemb = (
#             posemb[:, :num_prefix_tokens],
#             posemb[:, num_prefix_tokens:],
#         )
#     else:
#         posemb_prefix, posemb = None, posemb

#     # do the interpolation
#     embed_dim = posemb.shape[-1]
#     orig_dtype = posemb.dtype
#     posemb = posemb.float()  # interpolate needs float32
#     posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)  # type: ignore
#     posemb = F.interpolate(
#         posemb, size=new_size, mode=interpolation, antialias=antialias
#     )
#     posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
#     posemb = posemb.to(orig_dtype)

#     # add back extra (class, etc) prefix tokens
#     if posemb_prefix is not None:
#         posemb = torch.cat([posemb_prefix, posemb], dim=1)

#     if not torch.jit.is_scripting() and verbose:  # type: ignore
#         logging.info(f"Resized position embedding: {old_size} to {new_size}.")

#     return posemb


# def nchw_to(x: torch.Tensor, fmt: Format):
#     if fmt == Format.NHWC:
#         x = x.permute(0, 2, 3, 1)
#     elif fmt == Format.NLC:
#         x = x.flatten(2).transpose(1, 2)
#     elif fmt == Format.NCL:
#         x = x.flatten(2)
#     return x


# class PatchEmbed(nn.Module):
#     """2D Image to Patch Embedding"""

#     output_fmt: Format
#     dynamic_img_pad: torch.jit.Final[bool]  # type: ignore

#     def __init__(
#         self,
#         patch_size: int,
#         in_chans: int = 3,
#         embed_dim: int = 768,
#         norm_layer: Optional[Callable] = None,
#         flatten: bool = True,
#         output_fmt: Optional[str] = None,
#         bias: bool = True,
#     ):
#         super().__init__()
#         self.patch_size = python.to_2tuple(patch_size)

#         if output_fmt is not None:
#             self.flatten = False
#             self.output_fmt = Format(output_fmt)
#         else:
#             # flatten spatial dim and transpose to channels last, kept for bwd compat
#             self.flatten = flatten
#             self.output_fmt = Format.NCHW

#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
#         )
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)

#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

#         elif self.output_fmt != Format.NCHW:
#             x = nchw_to(x, self.output_fmt)

#         x = self.norm(x)
#         return x


# class PatchPosEmbed(nn.Module):
#     def __init__(
#         self,
#         patch_size,
#         num_patches,
#         in_chans,
#         embed_dim,
#     ):
#         super().__init__()
#         self.num_patches = num_patches

#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#         )

#         self.pos_param = nn.parameter.Parameter(torch.zeros(1, num_patches, embed_dim))

#     def pos_embed(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.pos_param
#         return x

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = self.pos_embed(x)
#         return x


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class PatchPosEmb(nn.Module):
    def __init__(
        self, input_channels: int, embed_dim: int, patch_size: int, num_patches: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_layer = nn.Linear(input_channels * (patch_size**2), embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # type: ignore
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))  # type: ignore

    def forward(self, x):
        # Encoder
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        return x


# if __name__ == "__main__":
#     patch_pos_embed = PatchPosEmbed(
#         num_patches=14 * 14,
#         patch_size=16,
#         in_chans=3,
#         embed_dim=768,
#     )

#     x = torch.randn(1, 3, 224, 224)
#     patch_pos_embed(x)
