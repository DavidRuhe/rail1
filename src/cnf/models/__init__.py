from .autoencoder import Autoencoder
from .cnf import ConditionalNeuralField
from .modules.local_decoder import LocalDecoder
from .modules.dgcnn_cls import DGCNN_cls
from . import convdfnet
def mnist_autoencoder():
    return Autoencoder(784, (512, 256, 128))


def mnist_field_decoder():
    return ConditionalNeuralField(
        input_dim=2,
        output_dim=1,
        hidden_dims=(128,) * 8,
        input_conditioning_dim=128,
        conditioning_hidden_dim=128,
    )


def shapenet_dfnet(c_dim, padding, encoder: dict, decoder: dict):

    dec = LocalDecoder(c_dim=c_dim, padding=padding, **decoder)

    # if name == "idx":
    #     raise NotImplementedError
    #     encoder = nn.Embedding(len(dataset), c_dim)
    # elif name is not None:
    enc = DGCNN_cls(c_dim, padding=padding, **encoder)
    # else:
    #     raise NotImplementedError
    #     encoder = None

    model = convdfnet.ConvolutionalDFNetwork(dec, enc)

    return model
