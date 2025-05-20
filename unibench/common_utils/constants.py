"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path
import os

##################################################################
# DIRECTORIES
##################################################################
PROJ_DIR = Path(__file__).parent.parent.absolute()
CURRENT_DIR = Path(os.getcwd())
HUB_CACHE_DIR = Path(os.getenv("TORCH_HOME", Path.home().joinpath(".cache").joinpath("torch"))).joinpath("hub")
CACHE_DIR = Path(os.getenv("UNIBENCH_HUB", Path.home().joinpath(".cache").joinpath("unibench")))

DATA_DIR = CACHE_DIR.joinpath("data")
OUTPUT_DIR = CACHE_DIR.joinpath("outputs")
LOCK_DIR = CACHE_DIR.joinpath("locks")

##################################################################
# MEAN AND STD
##################################################################

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = 'center'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
