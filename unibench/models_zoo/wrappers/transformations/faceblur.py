"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import torch
import torchvision


class FaceBlur(torch.nn.Module):
    def __init__(self, input_resolution=224, kernel_size=21, sigma=10.0):
        super(FaceBlur, self).__init__()
        self.mtcnn = MTCNN(keep_all=True, image_size=input_resolution)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, image):
        boxes, _ = self.mtcnn.detect(image)
        image_c = np.array(image.copy())
        if boxes is not None:
            for x, y, w, h in boxes:
                x, y, w, h = int(x), int(y), int(w), int(h)

                roi = image_c[y:h, x:w]

                if all([x >= 10 for x in roi.shape]):
                    continue

                roi = torchvision.transforms.functional.gaussian_blur(
                    Image.fromarray(roi), kernel_size=self.kernel_size, sigma=self.sigma
                )
                image_c[y:h, x:w] = roi
        return Image.fromarray(image_c)