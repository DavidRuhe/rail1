from .autoencoder import Autoencoder
from .cnf import ConditionalNeuralField
from .modules.local_decoder import LocalDecoder
from .modules.dgcnn_cls import DGCNN_cls
from . import s2vs_ae
from . import convdfnet
from .points_ae import RandomPointsAE


def mnist_autoencoder():
    return Autoencoder(784, (512, 256, 128))


def random_points_ae(n_points=1):
    return RandomPointsAE(n_points, (3 + n_points) ** 2, (512, 512, 384))


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


def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, deterministic=False):
    if deterministic:
        raise NotImplementedError
        model = AutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
        )
    else:
        model = s2vs_ae.KLAutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
        )
    return model


def s2vs_autoencoder(point_cloud_size):

    return create_autoencoder(
        dim=512, M=512, latent_dim=8, N=point_cloud_size, deterministic=False
    )
