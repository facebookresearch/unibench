"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch

class GrayScale2RGB(torch.nn.Module):
    def __init__(self):
        super(GrayScale2RGB, self).__init__()

    def forward(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image