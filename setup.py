#!/usr/bin/env python3
import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

setup(
    name="fieldExtractor",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "transformers==2.9.0",
        "tensorboardX==2.0",
        "lxml==4.6.5",
        "Pillow==9.3.0",
    ],
)
