"""
xvdp 2019
    extract of nms and roi from maskrcnn-benchmark
    for installation convenience

tested with ubuntu 16 and 18 only
"""

import os
import glob
import subprocess
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def set_version(version):

    with open('nms_pytorch/version.py', 'w') as _fi:
        _fi.write("version='"+version+"'")
    return version

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "nms_pytorch", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "nms_pytorch._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


ARGS = dict(
    name="nms_pytorch",
    version=set_version(version='0.0.5'),
    author="xvdp",
    description="nms from maskrcnn_benchmark",
    #install_requires=['torch', 'torchvision'],
    packages=find_packages(exclude=("tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    zip_safe=False,
)

try:
    setup(**ARGS)
except subprocess.CalledProcessError:
    print('Failed to build extension!')
    del ARGS['ext_modules']
    setup(**ARGS)
