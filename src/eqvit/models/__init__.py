from torch import nn
from .transformer import Transformer
from .vision_transformer import VisionTransformer

from . import processor
from . import decoder
from . import encoder
from . import epd


def reverse_transformer():
    return Transformer(
        input_dim=10, model_dim=32, num_heads=1, num_classes=10, num_layers=1, dropout=0
    )


def anomaly_transformer():
    return Transformer(
        input_dim=512,
        model_dim=256,
        num_heads=4,
        num_classes=1,
        num_layers=4,
        dropout=0.1,
        input_dropout=0.1,
    )


# def mnist_epd():
#     depth = 12

#     enc = encoder.PatchPosEmbed(
#         patch_size=4,
#         num_patches=7 * 7,
#         in_chans=1,
#         embed_dim=128,
#     )

#     proc = processor.Processor(
#         num_cores=16,
#         core_dim=128,
#         depth=depth,
#         num_heads=4,
#         cross_layers=set(range(depth)),
#     )

#     dec = decoder.Decoder(
#         input_dim=128,
#         output_dim=10,
#         pre_norm=True,
#         drop_rate=0.0,
#         global_pool="avg",
#     )

#     return epd.EncoderProcessorDecoder(enc, proc, dec)


def cifar10_epd():
    vit = epd.EPD(
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=4,
        num_channels=3,
        num_patches=64,
        num_classes=10,
        dropout=0.2,
    )
    return vit