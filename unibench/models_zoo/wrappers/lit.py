"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from common_utils.constants import CACHE_DIR
from .base import AbstractModel
import tensorflow as tf
import tqdm


class LiTModel(AbstractModel):
    def __init__(self, model, model_name, *args, **kwargs):
        super(LiTModel, self).__init__(model, model_name, *args, **kwargs)
        self.tokenizer = model.get_tokenizer()

    def _get_zeroshot_weights(self, class_names, templates):
        zeroshot_weights = []
        for class_name in tqdm(class_names):
            texts = [
                template.format(class_name) for template in self.templates
            ]  # format with class
            texts = self.tokenize(texts)  # tokenize
            _, class_embeddings, _ = self.model.apply(
                self.lit_variables, tokens=texts
            )  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

        return zeroshot_weights

    def get_zeroshot_predictions(self, images, zeroshot_weights):
        if self.zeroshot_weights is None:
            self.zeroshot_weights = self._get_zeroshot_weights(
                imagenet_classes, imagenet_templates
            )
        return self.get_image_embeddings(images) @ zeroshot_weights

    def get_image_embeddings(self, images):
        image_features = self.model.module.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_embeddings(self, text):
        caption_options = []
        for c_option in text:
            caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
            caption_embeddings = self.model.module.encode_text(
                caption_tokenized.to("cuda")
            )  # B x D
            caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)  # B x D
            caption_options.append(caption_embeddings.unsqueeze(1))  # B x 1 x D

        return torch.stack(caption_options, axis=1)

    def forward_batch(self, images, output="zero_shot", text=None):
        assert output in self.OUTPUT_TYPES

        if output == "zero_shot":
            return self.get_zeroshot_predictions(images, self.zeroshot_weights)
        elif output == "relations":
            image_options = self.get_image_embeddings(images)  # B x L x D
            caption_options = self.get_text_embeddings(text)  # B x K x D
            return np.einsum(
                "nkd,nld->nkl", image_options, caption_options
            )  # B x K x L
        raise NotImplementedError(f"Not implemented for {self.model_name}")
