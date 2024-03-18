import torch
from torch import nn
from torch.nn import init

class PoolingLayer(nn.Module):

    def __init__(self, input_channels, output_dim):

        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        self.weight_mlp = nn.Linear(input_channels, output_dim)
        self.
    
    def forward(self, input):
        
        w = self.weight_mlp(input).transpose(1, 2)
        c = (w @ input) / input.size(1) ** 0.5

        # Extension 1: K-means.
        # Extension 2: Do self-attention from here.

        B, N, _ = input.size()
        B, C, _ = c.size()

        h = torch.cat([
            torch.einsum("bci,bni->bcn", c, input),
            torch.square(c).sum(dim=-1)[:, None].expand((B, N, C)),
            torch.square(input).sum(dim=-1, keepdim=True).expand((B, N, C))
        ])


    


if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    input = torch.randn(1, 1024, 3)
    layer = PoolingLayer(3, 512)
    output = layer(input)
    plt.imshow(output.squeeze().detach().numpy())
    plt.show()
    breakpoint()
    print("done")