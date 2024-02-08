from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lltm",
    ext_modules=[
        # CUDAExtension(
        #     "lltm_cuda",
        #     [
        #         "lltm_cuda.cpp",
        #         "lltm_cuda_kernel.cu",
        #     ],
        # ),
        # CUDAExtension(
        #     "add_cuda",
        #     [
        #         "add_cuda.cpp",
        #         "add_cuda_kernel.cu",
        #     ],
        # ),
        # CUDAExtension(
        #     "dot_cuda",
        #     [
        #         "dot_cuda.cpp",
        #         "dot_cuda_kernel.cu",
        #     ],
        # ),
        CUDAExtension(
            "ip_cuda",
            [
                "ip_cuda.cpp",
                "ip_cuda_kernel.cu",
            ],
        ),

    ],
    cmdclass={"build_ext": BuildExtension},
)
