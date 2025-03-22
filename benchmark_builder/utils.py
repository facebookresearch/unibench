"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

###### CREDIT GOES TO https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets ######

import io

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