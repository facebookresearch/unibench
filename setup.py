"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="unibench",
    version="0.4.0",
    author="Haider Al-Tahan",
    author_email="haideraltahan@meta.com",
    description="This repository is designed to simplify the evaluation process of vision-language models. It provides a comprehensive set of tools and scripts for evaluating VLM models and benchmarks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/unibench",
    project_urls={
        "Bug Tracker": "https://github.com/facebookresearch/unibench/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "open-clip-torch",
        "openai-clip",
        "zipp",
        "timm",
        "fire",
        "opencv-python",
        "datasets",
        "ftfy",
        "torch",
        "scipy",
        "fairscale",
        "GitPython",
        "torchvision",
        "rich",
        "oslo.concurrency",
        "transformers",
        "gdown",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    entry_points={"console_scripts": ["unibench = unibench.evaluator:run"]},
    dependency_links=["https://pypi.nvidia.com"],
)
