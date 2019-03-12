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

    with open('torchvision_extra/version.py', 'w') as _fi:
        _fi.write("version='"+version+"'")
    return version

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchvision_extra", "csrc")

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

    print("sources:", sources)
    include_dirs = [extensions_dir]
    print("extensions_dir:", extensions_dir)
    print("main_file:", main_file)


    ext_modules = [
        extension(
            "torchvision_extra._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="torchvision_extra",
    version=set_version(version='0.0.9'),
    author="xvdp",
    description="extra layers to torchvision, including nms from maskrcnn_benchmark",
    #install_requires=['torch', 'torchvision'],
    packages=find_packages(exclude=("tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension}
)
