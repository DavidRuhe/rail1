# import math

# import torch
# from torch import nn

# from .linear import MVLinear
# from .normalization import NormalizationLayer

# import torch
# from torch import nn

# from .utils import unsqueeze_like


# class MVSiLU(nn.Module):
#     def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
#         super().__init__()
#         self.algebra = algebra
#         self.channels = channels
#         self.exclude_dual = exclude_dual
#         self.invariant = invariant
#         self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
#         self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))

#         if invariant == "norm":
#             self._get_invariants = self._norms_except_scalar
#         elif invariant == "mag2":
#             self._get_invariants = self._mag2s_except_scalar
#         else:
#             raise ValueError(f"Invariant {invariant} not recognized.")

#     def _norms_except_scalar(self, input):
#         return self.algebra.norms(input, grades=self.algebra.grades[1:])

#     def _mag2s_except_scalar(self, input):
#         return self.algebra.qs(input, grades=self.algebra.grades[1:])

#     def forward(self, input):
#         norms = self._get_invariants(input)
#         norms = torch.cat([input[..., :1], *norms], dim=-1)
#         a = unsqueeze_like(self.a, norms, dim=2)
#         b = unsqueeze_like(self.b, norms, dim=2)
#         norms = a * norms + b
#         norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
#         return torch.sigmoid(norms) * input

# class SteerableGeometricProductLayer(nn.Module):
#     def __init__(
#         self, algebra, features, include_first_order=True, normalization_init=0
#     ):
#         super().__init__()

#         self.algebra = algebra
#         self.features = features
#         self.include_first_order = include_first_order

#         if normalization_init is not None:
#             self.normalization = NormalizationLayer(
#                 algebra, features, normalization_init
#             )
#         else:
#             self.normalization = nn.Identity()
#         self.linear_right = MVLinear(algebra, features, features, bias=False)
#         if include_first_order:
#             self.linear_left = MVLinear(algebra, features, features, bias=True)

#         self.product_paths = algebra.geometric_product_paths
#         self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

#     def _get_weight(self):
#         weight = torch.zeros(
#             self.features,
#             *self.product_paths.size(),
#             dtype=self.weight.dtype,
#             device=self.weight.device,
#         )
#         weight[:, self.product_paths] = self.weight
#         subspaces = self.algebra.subspaces
#         weight_repeated = (
#             weight.repeat_interleave(subspaces, dim=-3)
#             .repeat_interleave(subspaces, dim=-2)
#             .repeat_interleave(subspaces, dim=-1)
#         )
#         return self.algebra.cayley * weight_repeated

#     def forward(self, input):
#         input_right = self.linear_right(input)
#         input_right = self.normalization(input_right)

#         weight = self._get_weight()

#         if self.include_first_order:
#             return (
#                 self.linear_left(input)
#                 + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
#             ) / math.sqrt(2)

#         else:
#             return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
