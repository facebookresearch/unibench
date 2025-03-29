"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from os.path import join as pjoin
from pathlib import Path
import os
import torch

##################################################################
# DIRECTORIES
##################################################################
PROJ_DIR = Path(__file__).parent.parent.absolute()
CURRENT_DIR = Path(os.getcwd())
HUB_CACHE_DIR = Path(os.getenv("TORCH_HOME", Path.home().joinpath(".cache").joinpath("torch"))).joinpath("hub")
CACHE_DIR = Path(os.getenv("UNIBENCH_HUB", Path.home().joinpath(".cache").joinpath("unibench")))
torch.hub.set_dir(HUB_CACHE_DIR)

DATA_DIR = CACHE_DIR.joinpath("data")
OUTPUT_DIR = CACHE_DIR.joinpath("outputs")
LOCK_DIR = CACHE_DIR.joinpath("locks")
