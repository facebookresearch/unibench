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
        max_new_tokens=32,
        image_token=None,
        **kwargs,
    ):
        kwargs['device'] = None
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

        inputs = (
            self.processor(
                text=prompts,
                padding=True,
                return_tensors="pt",
                images=[[image] for image in (images * 255).int()],
            )
        ).to(self.model.device).to(self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
        res = []
        for i, text in enumerate(gen_res):
            p = prompts[i]
            if self.image_token is not None:
                p = prompts[i].replace(self.image_token, '')
            res.append(self.output_func(text.split(p)[-1]))
        return res


class PaliGemma(VLLModel):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        res = []
        for image, prompt in zip(images, prompts):
            if self.processor.chat_template is not None:
                prompt = self.processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    add_generation_prompt=True,
                )
            elif self.image_token is not None:
                prompt = self.image_token + prompt
            inputs = (
                self.processor(
                    text=prompt,
                    images=[(image * 255).int()],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            ).to(self.model.device).to(self.model.dtype)
            if len(inputs['pixel_values'].shape) > 4:
                inputs['pixel_values'] = inputs['pixel_values'].squeeze()
                
            if 'image_sizes' in inputs:
                del inputs['image_sizes']
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
            p = prompt
            if self.image_token is not None:
                p = prompt.replace(self.image_token, '')
            res.append(self.output_func(gen_res[0].split(p)[-1]))

        return res
