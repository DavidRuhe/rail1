from .autoencoder import Autoencoder
from .cnf import ConditionalNeuralField
from .modules.local_decoder import LocalDecoder
from .modules.dgcnn_cls import DGCNN_cls
from . import s2vs_ae
from . import convdfnet
from .points_ae import RandomPointsAE, RandomSurfacesMLP
from .vn_dcgnn import VNDGCNN
from .pointnetpp import PointNetPPClassification
from .fpsnet import FPSNet
from .transposenet import TransposeNet
from .pointnet import PointNet
from .pointmlp import pointMLP
from .pointmlp_clean import construct_pointmlp
from .menghao_trafo import PointCloudTransformer
from .menghao_trafo_clean import PointTransformerClsClean
from .hengshuang_trafo import PointTransformerCls as HSPointTransformerCls
from .hengshuang_trafo_new import PointTransformerCls as HSPointTransformerClsNew
from .random_spheres_cnf import RandomSpheresCNF
from .fnf import CFNF
from .spheres_fnf import SpheresFNF, spheres_fnf, shapes_fnf, conditional_pointnet_fnf
from .s2vs import ae_d128_m512, ae_d512_m512, ae_d512_m128, ae_d128_m128


def cfnf(*args, **kwargs):
    return CFNF(*args, **kwargs)


def pointnet_siren(*args, **kwargs):
    return SCNF(backbone='pointnet', *args, **kwargs)

def cnf(*args, **kwargs):
    return ConditionalNeuralField(*args, **kwargs)

def siren_cnf(*args, **kwargs):
    return SirenCNF(*args, **kwargs)

def pointnet_cnf():
    return RandomSpheresCNF(backbone="pointnet")

def trafo_hengshuang_new():
    return HSPointTransformerClsNew()

def trafo_hengshuang():
    return HSPointTransformerCls()

def trafo_menghao():
    return PointCloudTransformer()

def trafo_menghao_clean():
    return PointTransformerClsClean()

def pointmlp_clean():
    return construct_pointmlp()


def pointmlp():
    return pointMLP()

def transposenet():
    return TransposeNet()

def fpsnet():
    return FPSNet()

def pointnetpp():
    return PointNetPPClassification()

def pointnetpp_kmeans():
    return PointNetPPClassification(kmeans=True)

def pointnet(*args, **kwargs):
    return PointNet(*args, **kwargs)

def vn_dgcnn():
    return VNDGCNN()

def mnist_autoencoder():
    return Autoencoder(784, (512, 256, 128))


def random_points_ae(num_points=1):
    return RandomPointsAE(num_points, (3 + num_points) ** 2, (512, 512, 384))

def rso_baseline(num_points=1):
    return RandomSurfacesMLP(num_points, 3, (512, 256, 128, 1))


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
