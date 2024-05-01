import torch
import torch.nn as nn


class FourierSeriesEmbedding(nn.Module):
    """
    Fourier series expansion of the input features.

    Args:
        in_dim (int): The number of input features.
        embedding_dim (int): Number of fourier coefficients to use.
    References:
        - See https://mathoverflow.net/questions/417033/fast-decaying-fourier-coefficients-for-indicator-function
    """

    def __init__(self, in_dim, embedding_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        # self.coefficients = nn.Parameter(torch.randn(embedding_dim) , requires_grad=True)
        self.coefficients = nn.Parameter(torch.arange(1, embedding_dim // 2 + 1, dtype=torch.float32) * 2 * torch.pi, requires_grad=False)
        # self.biases = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)
        # self.prenorm = nn.LayerNorm(in_dim)
        self.prenorm= nn.LayerNorm(in_dim)
        self.prenorm = nn.Identity()
        # self.postnorm = nn.BatchNorm1d(embedding_dim)
        self.postnorm = nn.LayerNorm(in_dim)
        self.postnorm = nn.Identity()
        # self.postnorm = nn.InstanceNorm1d(embedding_dim)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (...)
        Returns:
            torch.Tensor: shape (..., embedding_dim)
        """
        # x = x * self.embedding_dim * 2 * torch.pi
        # x = self.prenorm(x.transpose(1, 2)).transpose(1, 2)
        # tanh, correct instance norm.
        # x = self.prenorm(x)
        B, N, C = x.shape
        x = self.prenorm(x.view(-1, self.in_dim))
        x_sin = torch.sin(x[..., None] * self.coefficients)
        x_cos = torch.cos(x[..., None] * self.coefficients)
        x = torch.cat([x_sin, x_cos], dim=-1)


        x = x.reshape(B * N, self.embedding_dim, -1)

        # breakpoint()

        # Batchnorm transpose works roughly.
        # Normal batchnorm roughly equal (slightly worse)
        # Layernorm best so far.
        # [ ] Layernorm Transpose worse
        # different scaling.
        # Check grad norm with scaling 0
        # x = self.postnorm(x)
        # x = self.postnorm(x.transpose(-2, -1)).transpose(-2, -1)

        x = x * 1 / (self.embedding_dim * torch.pi)

        x = x.reshape(B, N, -1)
        return x




class SigmoidEmbedding(nn.Module):

    def __init__(self, in_dim, embedding_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.coefficients = nn.Parameter(torch.randn(embedding_dim) * 10, requires_grad=True)
        self.biases = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, x):

        x = (x[..., None] - self.biases).square() * self.coefficients.exp()

        return (-x).exp()