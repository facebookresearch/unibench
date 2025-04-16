"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from .base import AbstractModel


class VLLModel(AbstractModel):
    def __init__(
        self,
        model,
        model_name,
        processor=None,
        output_func=None,
        max_new_tokens=16,
        image_token=None,
        **kwargs,
    ):
        super(VLLModel, self).__init__(
            model, model_name, use_transforms=False, **kwargs
        )
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.output_func = output_func
        self.image_token = image_token

    def get_text_from_image(self, images, prompts):
        pass

    def get_image_embeddings(self, images):
        pass

    def get_text_embeddings(self, texts):
        pass

    def compute_zeroshot_weights(self):
        pass


class LlavaModels(VLLModel):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        for i in range(len(prompts)):
            if self.processor.chat_template is not None:
                prompts[i] = self.processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompts[i]},
                            ],
                        },
                    ],
                    add_generation_prompt=True,
                )
            elif self.image_token is not None:
                prompts[i] = self.image_token + prompts[i]

        inputs = self.processor(
            text=prompts, padding=True, return_tensors="pt", images=(images * 255).int()
        ).to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
        res = []
        for text in gen_res:
            res.append(self.output_func(text))
        return res


class PaliGemma(VLLModel):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        res = []
        for image, prompt in zip(images, prompts):
            inputs = self.processor(
                text=[self.image_token + prompt if self.image_token is not None else prompt],
                images=(image * 255).int(),
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
            res.append(self.output_func(gen_res[0].split(prompt)[-1]))

        return res
