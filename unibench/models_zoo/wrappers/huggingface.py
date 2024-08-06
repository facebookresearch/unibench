"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch

from .blip import BlipModel
from .clip import ClipModel


class FlavaModel(ClipModel):
    def __init__(self, *args, **kwargs):
        super(FlavaModel, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def get_image_embeddings(self, images):
        image_features = self.model.flava.get_image_features(
            images.to(self.device),
        )[:, 0, :]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.unsqueeze(1)

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        captions = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.context_length,
            truncation=True
        )

        text_features = self.model.flava.get_text_features(
            **{k: v.cuda() for k, v in captions.items()}
        )[:, 0, :]
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features


class XVLMModel(BlipModel):
    def __init__(self, *args, **kwargs):
        super(XVLMModel, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def get_image_embeddings(self, images):
        image_features_out = self.model.vision_encoder(
            images.to(self.device),
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        image_features = self.model.vision_proj(image_features_out[:, 0, :])
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.unsqueeze(1)
