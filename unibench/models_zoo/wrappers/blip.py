"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import itertools
import torch

from .clip import ClipModel


class BlipModel(ClipModel):
    def __init__(self, **kwargs):
        super(BlipModel, self).__init__(**kwargs)

    @torch.no_grad()
    def get_image_embeddings(self, images):
        image_features = self.model.visual_encoder(images.to(self.device))
        image_features = self.model.vision_proj(image_features[:, 0, :]).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.unsqueeze(1)

    @torch.no_grad()
    def use_mlp_head(self, sim_scores, image_features, captions):
        num_captions = len(captions)
        batch_size = len(captions[0])
        captions = self.tokenizer(
            list(itertools.chain.from_iterable(captions)),
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )

        text_token = (
            captions["input_ids"]
            .to(self.device)
            .reshape(num_captions, batch_size, -1)
            .permute(1, 0, 2)
        )
        text_attention = (
            captions["attention_mask"]
            .to(self.device)
            .reshape(num_captions, batch_size, -1)
            .permute(1, 0, 2)
        )

        # AUGMENT TEXT to IMG scores by parsing the text conditioned on the image
        new_scores = torch.full(sim_scores.size(), -100.0).to(
            self.device
        )  # batch x n_image_options x n_text_options)
        n_text_options = new_scores.size(2)
        for i in range(sim_scores.size(0)):
            text_candidates_i = text_token[i]
            text_attention_i = text_attention[i]
            encoder_att = torch.ones(
                (n_text_options, image_features.size(2)), dtype=torch.long
            ).to(self.device)
            for j in range(sim_scores.size(1)):  # loop over image options
                encoder_output = image_features[i, j]  # size n hidden states x dim d
                encoder_output = encoder_output.repeat(n_text_options, 1, 1)
                output = self.model.text_encoder(
                    text_candidates_i,
                    attention_mask=text_attention_i,
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=False,
                )[0]
                score = self.model.itm_head(output[:, 0, :])[
                    :, 1
                ]  # logits that the text is relevant to the image
                new_scores[i, j, :] = score + sim_scores[i, j, :]
        return new_scores

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        captions = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )

        text_features = self.model.text_encoder(
            captions["input_ids"].to(self.device),
            attention_mask=captions["attention_mask"].to(self.device),
            return_dict=True,
            mode="text",
        )

        text_features = self.model.text_proj(
            text_features.last_hidden_state[:, 0, :]
        ).float()

        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
