#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os

import torch
import torch.utils.cpp_extension as cpp_extension
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


requirements = [
    "torchvision",
    "ninja",
    "yacs",
    "cython",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "scikit-image"
]

if torch.version.hip is not None and CUDA_HOME is None:
    cpp_extension.CUDA_HOME = "/usr"
    CUDA_HOME = cpp_extension.CUDA_HOME


def get_extensions():
    extensions_dir = os.path.join("maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    has_accelerator = torch.cuda.is_available() and CUDA_HOME is not None
    force_accelerator = os.getenv("FORCE_CUDA", "0") == "1"
    if has_accelerator or force_accelerator:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        if torch.version.hip is None:
            extra_compile_args["nvcc"] = [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
