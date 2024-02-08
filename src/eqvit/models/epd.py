from torch import nn
from .vision_transformer import Encoder
from .processor import Processor
from .vision_transformer import Decoder


class EPD(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.encoder = Encoder(num_channels, patch_size, embed_dim, num_patches)
        self.processor = Processor(
            64,
            embed_dim,
            num_heads,
            depth=num_layers,
            mlp_dropout=dropout,
            attn_dropout=dropout,
            core_dropout=dropout,
            input_dropout=dropout,
        )
        self.decoder = Decoder(embed_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        x = self.decoder(x)
        return x
