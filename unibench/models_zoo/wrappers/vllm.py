"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from .base import AbstractModel
import torch._dynamo
from torchvision.transforms import functional as F

class AbstractVLLM(AbstractModel):
    def __init__(
        self,
        model,
        model_name,
        processor=None,
        output_func=None,
        inp_processor_func=None,
        generation_config={},
        gen_kwargs={},
        max_new_tokens=32,
        image_token=None,
        use_img_size=False,
        **kwargs,
    ):
        kwargs["device"] = None
        super(AbstractVLLM, self).__init__(
            model, model_name, use_transforms=False, **kwargs
        )
        self.use_img_size = use_img_size
        self.processor = processor
        self.inp_processor_func = inp_processor_func
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = gen_kwargs
        self.generation_config=generation_config
        self.output_func = output_func
        self.image_token = image_token
        self.model = torch.compile(self.model, dynamic=False)
        torch._dynamo.config.recompile_limit = 64 

    def get_text_from_image(self, images, prompts):
        pass

    def get_image_embeddings(self, images):
        pass

    def get_text_embeddings(self, texts):
        pass

    def compute_zeroshot_weights(self):
        pass


class VLLModels(AbstractVLLM):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        prompts = list(prompts)
        images = images.clone()
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
            (
                self.processor(
                    text=prompts,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    images=[[image] for image in (images * 255).int()],
                    pad_to_multiple_of=8
                )
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )
        
        if "image_sizes" in inputs and not self.use_img_size:
            del inputs["image_sizes"]
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, **self.gen_kwargs)
        output = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
        res = []
        for i, text in enumerate(gen_res):
            res.append(self.output_func(text))
        return res
    

class InternVLModels(VLLModels):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        prompts = list(prompts)
        images = images.clone()

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)

        for i in range(len(prompts)):
            prompts[i] = self.image_token + prompts[i]

        num_patches_list = [1 for _ in range(len(images))]

        gen_res = self.model.batch_chat(self.processor, images.bfloat16().to(self.model.device), num_patches_list=num_patches_list, questions=prompts, generation_config=generation_config)

        res = []
        for i, text in enumerate(gen_res):
            p = prompts[i]
            if self.image_token is not None:
                p = prompts[i].replace(self.image_token, "")
            res.append(self.output_func(text.split(p)[-1]))
        return res
    
class PHIModels(VLLModels):
    @torch.no_grad()
    def get_text_from_image(self, images, prompts):
        prompts = list(prompts)
        images = images.clone()
        for i in range(len(prompts)):
            prompts[i] = self.inp_processor_func(prompts[i])
                

        inputs = (
            (
                self.processor(
                    text=prompts,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    images=[F.to_pil_image(image) for image in images],
                )
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )
        
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, generation_config=self.generation_config)
        output = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        gen_res = self.processor.batch_decode(output, skip_special_tokens=True)
        res = []
        for i, text in enumerate(gen_res):
            res.append(self.output_func(text))
        return res