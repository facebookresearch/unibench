"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from .base import AbstractModel
import inspect

import re
from transformers import CLIPVisionModel
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria
from nltk.tokenize import sent_tokenize, word_tokenize
import transformers

from transformers import AutoTokenizer


class LVLModel(AbstractModel):
    def __init__(
        self,
        model,
        model_name,
        prompt='Fill in the Blank: This is a photo of a {}.',
        processor=None,
        **kwargs,
    ):
        super(LVLModel, self).__init__(model, model_name, **kwargs)
        self.prompt = prompt
        self.processor = processor

        self.embedding_model = transformers.CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float16
        ).cuda()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.embedding_model.eval()

    def compute_zeroshot_weights(self):
        zeroshot_weights = []
        for class_name in self.classes:
            texts = [template.format(class_name) for template in self.templates]

            class_embedding = self.get_text_embeddings(texts)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights).T

    @torch.no_grad()
    def get_image_embeddings(self, images):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        
        if self.processor.chat_template is not None:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )            
        else:
            prompt = self.prompt

        prompts = [prompt] * len(images)
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = images.to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=77)
        gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
        res = []
        for text in gen_res:
            if self.processor.chat_template is not None:
                res.append(text.split("ASSISTANT:")[-1].strip().replace("\n", ""))
            else:
                res.append(text.split(prompt)[-1].strip().replace("\n", ""))

        return self.get_text_embeddings(res).unsqueeze(1)

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        text_descriptor = self.embedding_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].cuda()
        text_descriptor_embeds = self.embedding_model(text_descriptor).text_embeds
        text_descriptor_embeds = text_descriptor_embeds / text_descriptor_embeds.norm(
            p=2, dim=-1, keepdim=True
        )
        return text_descriptor_embeds


class LlavaNext(LVLModel):
    @torch.no_grad()
    def get_image_embeddings(self, images):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        
        if self.processor.chat_template is not None:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )            
        else:
            prompt = self.prompt
            
        res = []
        for image in images:
            inputs = self.processor(text=prompt, images=image,  padding=True, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=77)
            gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
            res.append(gen_res[0].split("ASSISTANT:")[-1].strip().replace("\n", ""))

        return self.get_text_embeddings(res).unsqueeze(1)
    
class PaliGemma(LVLModel):
    @torch.no_grad()
    def get_image_embeddings(self, images):            
        res = []
        for image in images:
            inputs = self.processor(text=[self.prompt], images=image,  padding=True, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=77)
            gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
            res.append(gen_res[0].split(self.prompt)[-1].strip().replace("\n", ""))

        return self.get_text_embeddings(res).unsqueeze(1)
    
    
class Chameleon(LVLModel):
    @torch.no_grad()
    def get_image_embeddings(self, images):            
        res = []
        for image in images:
            inputs = self.processor(text=[self.prompt], images=image,  padding=True, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=77)
            gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
            res.append(gen_res[0].split(self.prompt.replace('<image>', ''))[-1].strip().replace("\n", ""))

        return self.get_text_embeddings(res).unsqueeze(1)
    