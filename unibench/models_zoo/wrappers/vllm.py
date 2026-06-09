"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import torch
from .base import AbstractModel
import torch._dynamo
from torchvision.transforms import functional as F
from openai import OpenAI
import base64
from io import BytesIO

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
        if self.model is not None:
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
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
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
    
class Qwen35VLModels(AbstractVLLM):
    """Qwen3.5 VL models via OpenAI-compatible API (e.g. vLLM server)."""

    def __init__(self, model_name, api_model_id, base_url=None, api_key="EMPTY", output_func=None, max_new_tokens=32, **kwargs):
        super(Qwen35VLModels, self).__init__(model=None, model_name=model_name, output_func=output_func or (lambda x: x), max_new_tokens=max_new_tokens, **kwargs)
        self.api_model_id = api_model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _image_to_data_url(self, image_tensor):
        pil_image = F.to_pil_image(image_tensor.clamp(0, 1))
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def get_text_from_image(self, images, prompts):
        prompts = list(prompts)
        images = images.clone()
        res = []
        for image, prompt in zip(images, prompts):
            data_url = self._image_to_data_url(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            response = self.client.chat.completions.create(
                model=self.api_model_id,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0,
                top_p=None,
            )
            text = response.choices[0].message.content or ""
            res.append(self.output_func(text))
        return res


class ChatGPTModels(AbstractVLLM):
    """GPT-4o / ChatGPT models via the OpenAI Chat Completions API."""

    def __init__(
        self,
        model_name,
        api_model_id,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        output_func=None,
        max_new_tokens=32,
        system_prompt=None,
        **kwargs,
    ):
        super(ChatGPTModels, self).__init__(
            model=None,
            model_name=model_name,
            output_func=output_func or (lambda x: x),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self.api_model_id = api_model_id
        self.system_prompt = system_prompt
        # api_key defaults to None; the OpenAI client will automatically
        # read OPENAI_API_KEY from the environment when no key is supplied.
        self.client = OpenAI(api_key=api_key)

    def _image_to_data_url(self, image_tensor):
        pil_image = F.to_pil_image(image_tensor.clamp(0, 1))
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def get_text_from_image(self, images, prompts):
        prompts = list(prompts)
        images = images.clone()
        res = []
        for image, prompt in zip(images, prompts):
            data_url = self._image_to_data_url(image)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            response = self.client.chat.completions.create(
                model=self.api_model_id,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            res.append(self.output_func(text))
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
