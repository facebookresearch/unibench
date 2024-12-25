"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
from typing import List, Union

import torch
import math
import timm.data
import torch
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    Normalize,
    ToTensor,
    InterpolationMode,
)

from .transformations.grayscale2rgb import GrayScale2RGB


class AbstractModel(ABC):
    def __init__(
        self,
        model,
        model_name,
        input_resolution: int = 224,
        crop_pct: float = 1.0,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        use_norm: bool = True,
        batch_per_gpu: int = 32,
        logit_scale=None,
        context_length=77,
        tokenizer=None,
        interpolation=InterpolationMode.BICUBIC,
        use_itm_head: bool = False,
        face_blur: bool = False,
        device: str = "cuda",
    ) -> None:
        super(AbstractModel, self).__init__()
        assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"

        self.model = model
        self.use_itm_head = use_itm_head
        self.batch_per_gpu = batch_per_gpu
        self.logit_scale = logit_scale
        self.context_length = context_length
        self.model_name = model_name
        self.crop_pct = crop_pct
        self.input_resolution = input_resolution
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.interpolation = interpolation
        self.use_norm = use_norm
        self.face_blur = face_blur
        self.device = device
        self.tokenizer = tokenizer

        self.model = self.model.to(device)
        self.model.eval()

        self.zeroshot_weights = None
        self.classes = None
        self.templates = None

    @abstractmethod
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_text_embeddings(self, texts: Union[list, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def set_templates(self, templates: List[str]) -> None:
        if self.templates != templates:
            self.templates = templates

    def set_classes(self, classes: List[str]) -> None:
        if self.classes != classes:
            self.classes = classes

    def get_zeroshot_predictions(self, images, zeroshot_weights):
        return (
            (
                self.logit_scale.exp()
                if self.logit_scale is not None
                else torch.tensor(100.0)
            )
            * self.get_image_embeddings(images)
            @ zeroshot_weights
        ).squeeze()

    @abstractmethod
    def compute_zeroshot_weights(self) -> None:
        pass

    def get_batch_size(self) -> int:
        return self.batch_per_gpu * (
            1 if self.device == "cpu" else torch.cuda.device_count()
        )

    def get_preprocess_transforms(self):
        scale_size = int(math.floor(self.input_resolution / self.crop_pct))
        transforms = [
            Resize(scale_size, interpolation=self.interpolation),
            CenterCrop(self.input_resolution),
        ]

        if self.face_blur:
            from .transformations.faceblur import FaceBlur
            transforms.append(FaceBlur(input_resolution=self.input_resolution))

        transforms.append(ToTensor())
        transforms.append(GrayScale2RGB())

        if self.use_norm:
            transforms.append(
                Normalize(
                    mean=self.norm_mean,
                    std=self.norm_std,
                )
            )

        return Compose(transforms)
