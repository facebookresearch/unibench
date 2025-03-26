"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

###### CREDIT GOES TO https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets ######

import io
import torch


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def PIL_to_bytes(image_format):
    OPTIONS = {
        "webp": dict(format="webp", lossless=True),
        "png": dict(format="png"),
        "jpg": dict(format="jpeg"),
    }

    def transform(image):
        bytestream = io.BytesIO()
        image.save(bytestream, **OPTIONS[image_format])
        return bytestream.getvalue()

    return transform
