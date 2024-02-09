from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, dims=(256, 128, 64), activation=nn.GELU()):
        super().__init__()
        self.input_dim = input_dim
        self.dims = dims
        self.activation = activation

        encoder = [nn.Linear(input_dim, dims[0]), activation]
        for i in range(len(dims) - 1):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            encoder.append(activation)
        self.encoder = nn.Sequential(*encoder)

        decoder = []

        for i in range(len(dims) - 1, 0, -1):
            decoder.append(nn.Linear(dims[i], dims[i - 1]))
            decoder.append(activation)
        decoder.append(nn.Linear(dims[0], input_dim))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
