"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from unibench.models_zoo import register_model
from unibench.common_utils.constants import (
    HUB_CACHE_DIR,
    CURRENT_DIR,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava 1.5 7B",
        "year": 2023,
        "month": 9,  # September 2023 release
    },
)
def llava_1_5_7b(model_name, **kwargs):
    from transformers import LlavaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("ASSISTANT:")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava 1.5 7B",
        "year": 2023,
        "month": 9,  # September 2023 release
    },
)
def llava_1_5_13b(model_name, **kwargs):
    from transformers import LlavaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-1.5-13b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("ASSISTANT:")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next",
    },
)
def aya_vision_8b(model_name, **kwargs):
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "CohereLabs/aya-vision-8b"
    model = AutoModelForImageTextToText.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )

    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.split("<|CHATBOT_TOKEN|>")[-1]
        .strip()
        .replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next",
    },
)
def aya_vision_32b(model_name, **kwargs):
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "CohereLabs/aya-vision-32b"
    model = AutoModelForImageTextToText.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )

    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.split("<|CHATBOT_TOKEN|>")[-1]
        .strip()
        .replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_next_llama_8b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llama3-llava-next-8b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        use_img_size=True,
        output_func=lambda x: x.split("assistant")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llama_4_scout(model_name, **kwargs):
    from transformers import Llama4ForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    model = Llama4ForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size['width'],
        output_func=lambda x: x.split("assistant")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]

@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def phi_4(model_name, **kwargs):
    from transformers import AutoModelForCausalLM
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "microsoft/Phi-4-multimodal-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
        _attn_implementation='eager',
    )
    model.load_adapter(
        name,
        adapter_name="vision",
        device_map="balanced",
        adapter_kwargs={"subfolder": "vision-lora"},
    )
    model.set_adapter("vision")

    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("<|CHATBOT_TOKEN|>")[-1]
        .strip()
        .replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]

@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_34b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-v1.6-34b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        use_img_size=True,
        output_func=lambda x: x.split("assistant")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]

@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def gemma3_4b(model_name, **kwargs):
    from transformers import Gemma3ForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.split("\nmodel\n")[-1]
        .strip()
        .replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def gemma3_27b(model_name, **kwargs):
    from transformers import Gemma3ForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/gemma-3-27b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.split("\nmodel\n")[-1]
        .strip()
        .replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]

@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_72b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-next-72b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("assistant")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_110b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-next-110b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("assistant")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_mistral_7b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        use_img_size=True,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.split("[/INST]")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_vicuna_7b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-v1.6-vicuna-7b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )

    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        use_img_size=True,
        output_func=lambda x: x.split("ASSISTANT:")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava Next Llama 8B",
        "year": 2024,
        "month": 1,  # March 2024 release
    },
)
def llava_1_6_vicuna_13b(model_name, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "llava-hf/llava-v1.6-vicuna-13b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )

    return VLLModels(
        model=model,
        model_name=model_name,
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        use_img_size=True,
        output_func=lambda x: x.split("ASSISTANT:")[-1].strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Chameleon",
        "year": 2024,
        "month": 4,  # April 2024 release
    },
)
def chameleon_7b(model_name, **kwargs):
    from transformers import ChameleonForConditionalGeneration
    from transformers import ChameleonProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "facebook/chameleon-7b"
    model = ChameleonForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=False,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = ChameleonProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Chameleon",
        "year": 2024,
        "month": 4,  # April 2024 release
    },
)
def chameleon_30b(model_name, **kwargs):
    from transformers import ChameleonForConditionalGeneration
    from transformers import ChameleonProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "facebook/chameleon-30b"
    model = ChameleonForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=False,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = ChameleonProcessor.from_pretrained(
        name, use_fast=True, padding_side="left", torch_dtype=torch.bfloat16
    )
    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.crop_size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 5,  # May 2024 release
    },
)
def paligemma_3b_224(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 5,  # May 2024 release
    },
)
def paligemma_3b_448(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma-3b-pt-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma_3b_mix_224(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_3b_mix_224(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma_3b_mix_448(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma-3b-mix-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_3b_mix_448(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-3b-mix-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_10b_mix_224(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-10b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_10b_mix_448(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-10b-mix-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_28b_mix_448(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-28b-mix-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


@register_model(
    "vllm",
    {
        "dataset_size": 14,
        "model_size": 7000,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "Llava",
        "year": 2024,
        "month": 6,  # June 2024 release
    },
)
def paligemma2_28b_mix_224(model_name, **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import AutoProcessor
    import torch
    from unibench.models_zoo.wrappers.vllm import VLLModels

    name = "google/paligemma2-28b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(name, use_fast=True, padding_side="left")

    return VLLModels(
        model=model,
        model_name=model_name,
        image_token="<image>",
        processor=processor,
        norm_mean=processor.image_processor.image_mean,
        norm_std=processor.image_processor.image_std,
        input_resolution=processor.image_processor.size["width"],
        output_func=lambda x: x.strip().replace("\n", ""),
        **kwargs
    ), [
        "text_classification",
        "clip_judge_classification",
        "llm_judge_classification",
        "clip_judge_relation",
        "in_context_text_classification",
    ]


def load_blip(model_name, model_url, model_size="base", image_size=224, **kwargs):
    from git import Repo
    from unibench.models_zoo.wrappers import BlipModel

    if not HUB_CACHE_DIR.joinpath("BLIP").exists():
        Repo.clone_from(
            "https://github.com/salesforce/BLIP.git", HUB_CACHE_DIR.joinpath("BLIP")
        )
    sys.path.append(str(HUB_CACHE_DIR.joinpath("BLIP")))
    from models.blip_itm import blip_itm

    os.chdir(str(HUB_CACHE_DIR.joinpath("BLIP")))

    model = blip_itm(
        pretrained=model_url,
        image_size=image_size,
        vit=model_size,
    )
    os.chdir(str(CURRENT_DIR))

    return BlipModel(
        model=model,
        model_name=model_name,
        tokenizer=model.tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        use_itm_head=True,
        input_resolution=image_size,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 14,
        "model_size": 86,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT B 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitB16_14m(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth",
        model_size="base",
        image_size=224,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 307,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT L 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitL16_129m(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
        model_size="large",
        image_size=224,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 86,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT B 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitB16_129m(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
        model_size="base",
        image_size=224,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 86,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT B 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitB16_coco(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
        model_size="base",
        image_size=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 86,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT B 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitB16_flickr(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
        model_size="base",
        image_size=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 307,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT L 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitL16_coco(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth",
        model_size="large",
        image_size=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 129,
        "model_size": 307,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT L 16 ",
        "year": 2022,
        "month": 1,
    },
)
def blip_vitL16_flickr(model_name, **kwargs):
    return load_blip(
        model_name=model_name,
        model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth",
        model_size="large",
        image_size=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 4350,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT E 14",
        "year": 2023,
        "month": 3,
    },
)
def eva02_vitE14_plus_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-E-14-plus", pretrained="laion2b_s9b_b144k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 4350,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT E 14",
        "year": 2023,
        "month": 3,
    },
)
def eva02_vitE14_2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-E-14", pretrained="laion2b_s4b_b115k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-E-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT L 14",
        "year": 2023,
        "month": 3,
    },
)
def eva02_vitL14_2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-L-14", pretrained="merged2b_s4b_b131k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT B 16",
        "year": 2023,
        "month": 3,
    },
)
def eva02_vitB16_2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-B-16", pretrained="merged2b_s8b_b131k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1011,
        "learning_objective": "EVA01",
        "architecture": "vit",
        "name": "EVA01 ViT g 14",
        "year": 2022,
        "month": 11,
    },
)
def eva01_vitG14_plus_2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA01-g-14-plus", pretrained="merged2b_s11b_b114k"
    )

    tokenizer = open_clip.get_tokenizer("EVA01-g-14-plus")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 1011,
        "learning_objective": "EVA01",
        "architecture": "vit",
        "name": "EVA01 ViT g 14",
        "year": 2022,
        "month": 11,
    },
)
def eva01_vitG14_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "EVA01-g-14", pretrained="laion400m_s11b_b41k"
    )

    tokenizer = open_clip.get_tokenizer("EVA01-g-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 1843,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT G 14",
        "year": 2023,
        "month": 5,
    },
)
def clipa_vitbigG14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 22,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-S",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_s_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-S", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-S")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 22,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-S-LTT",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_s_ltt_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-S-LTT", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-S-LTT")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 87,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-B",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_b_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-B", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-B")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 87,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-B-LTT",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_b_ltt_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-B-LTT", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-B-LTT")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l2_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L2", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L2")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2-256",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l2_256_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L2-256", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2-336",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l2_336_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L2-336", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L2-336")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-256",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_xl_256_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-XL-256", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-XL-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-336",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_xl_336_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-XL-336", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-XL-336")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-384",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_xl_384_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-XL-384", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-XL-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L-256",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l_256_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L-256", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L-336",
        "year": 2024,
        "month": 4,
    },
)
def vitamin_l_336_1b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViTamin-L-336", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViTamin-L-336")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 633,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT H 14",
        "year": 2023,
        "month": 5,
    },
)
def clipa_vitH14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 307,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT L 14",
        "year": 2023,
        "month": 5,
    },
)
def clipa_vitL14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT L 16",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitL16(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "Roberta ViT B 32",
        "year": 2022,
        "month": 11,
    },
)
def roberta_vitB32(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "roberta-ViT-B-32", pretrained="laion2b_s12b_b32k"
    )

    tokenizer = open_clip.get_tokenizer("roberta-ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitB16(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 256",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitB16_256(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 384",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitB16_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 512",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitB16_512(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT L 16 384",
        "year": 2023,
        "month": 3,
    },
)
def siglip_vitL16_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14",
        "year": 2023,
        "month": 3,
    },
)
def siglip_so400_14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14 378",
        "year": 2023,
        "month": 3,
    },
)
def siglip_so400_14_378(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-378", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-378")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14 384",
        "year": 2023,
        "month": 3,
    },
)
def siglip_so400_14_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 512",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_16_512(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 512",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_16_512(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 384",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_16_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 256",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_16_256(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 14 378",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_14_378(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP2-378", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2-378")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 14",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_so400_14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP2", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 512",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitL16_512(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 384",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitL16_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 256",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitL16_256(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 512",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitB16_512(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 384",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitB16_384(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 256",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitB16_256(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitB16(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 32 256",
        "year": 2025,
        "month": 2,
    },
)
def siglip2_vitB32_256(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=IMAGENET_INCEPTION_MEAN,
        norm_std=IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 32",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitB32_metaclip_fullcc(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 16",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitB16_metaclip_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 32",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitB32_metaclip_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "ViT B 32 GeLU",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitB32_quickgelu_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "ViT B 32 GeLU",
        "year": 2021,
        "month": 1,
    },
)
def openclip_vitB32_quickgelu_openai(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 16",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitB16_metaclip_fullcc(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitL14_dfn2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="dfn2b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT L 14",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitL14_metaclip_400(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT L 14",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitL14_metaclip_fullcc(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT H 14",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitH14_metaclip_fullcc(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 5000,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT H 14",
        "year": 2023,
        "month": 9,
    },
)
def openclip_vitH14_dfn5b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="dfn5b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 88,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ConvNext",
        "year": 2021,
        "month": 7,
    },
)
def openclip_convnext_base(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_base", pretrained="laion400m_s13b_b51k"
    )

    tokenizer = open_clip.get_tokenizer("convnext_base")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "clipHero",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT B 32",
        "year": 2021,
        "month": 1,
    },
)
def clip_vitB32(model_name, **kwargs):
    import clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _ = clip.load("ViT-B/32", download_root=str(HUB_CACHE_DIR))

    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 13,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitB32_datacomp_s(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_s_s13m_b4k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 128,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitB32_datacomp_m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_m_s128m_b4k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitB32_datacomp_xl(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 16",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitB16_datacomp_xl(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 16",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitB16_datacomp_l(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_l_s1b_b8k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT H 14",
        "year": 2021,
        "month": 7,
    },
)
def openclip_vitH14(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


# @register_model(
#     "vision_text",
#     {
#         "dataset_size": 16,
#         "model_size": 86,
#         "learning_objective": "XVLM",
#         "architecture": "Swin",
#         "name": "XVLM Swin B",
#         "year": 2021,
#         "month": 11
#     },
# )
# def xvlm_flickr(model_name, **kwargs):
#     from unibench.models_zoo.wrappers import XVLMModel
#     from .wrappers.xvlm_util.xvlm import XVLM
#     from .wrappers.xvlm_util.tokenization_bert import BertTokenizer
#     from .wrappers.xvlm_util.tokenization_roberta import RobertaTokenizer
#     from .wrappers.xvlm_util.utils import get_config

#     config, model_path = get_config("xvlm-flickr")

#     model = XVLM(config)

#     model.load_pretrained(
#         model_path,
#         config,
#         is_eval=True,
#         is_pretrained=False,  # never used pretrained in NegCLIP paper?
#     )

#     if config["use_roberta"]:
#         tokenizer = RobertaTokenizer.from_pretrained(config["text_encoder"])
#     else:
#         # TODO: Hack. We should use the tokenizer from the config
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     return XVLMModel(
#         model=model,
#         model_name=model_name,
#         tokenizer=tokenizer,
#         norm_mean=OPENAI_CLIP_MEAN,
#         norm_std=OPENAI_CLIP_STD,
#         input_resolution=384,
#         **kwargs
#     ), [
#         "zeroshot_classification",
#         "zeroshot_relation",
#     ]


@register_model(
    "vision_text",
    {
        "dataset_size": 70,
        "model_size": 86,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "FLAVA ViT B 32",
        "year": 2021,
        "month": 12,
    },
)
def flava_full(model_name, **kwargs):
    from unibench.models_zoo.wrappers import FlavaModel
    from transformers import FlavaForPreTraining, FlavaImageProcessor, BertTokenizer

    model = FlavaForPreTraining.from_pretrained("facebook/flava-full")

    processor = FlavaImageProcessor.from_pretrained("facebook/flava-full")

    tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

    return FlavaModel(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=processor.size["height"],
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitL14_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT L 14",
        "year": 2023,
        "month": 4,
    },
)
def openclip_vitL14_datacomp_xl(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
        "year": 2021,
        "month": 7,
    },
)
def openclip_vitL14_2b(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT L 14",
        "year": 2021,
        "month": 1,
    },
)
def clip_vitL14(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("ViT-L/14", download_root=str(HUB_CACHE_DIR))

    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


# @register_model(
#     "vision_text",
#     {
#         "dataset_size": 16,
#         "model_size": 86,
#         "learning_objective": "XVLM",
#         "architecture": "Swin",
#         "name": "XVLM Swin B",
#         "year": 2021,
#         "month": 11
#     },
# )
# def xvlm_coco(model_name, **kwargs):
#     from unibench.models_zoo.wrappers import XVLMModel
#     from .wrappers.xvlm_util.xvlm import XVLM
#     from .wrappers.xvlm_util.tokenization_bert import BertTokenizer
#     from .wrappers.xvlm_util.tokenization_roberta import RobertaTokenizer
#     from .wrappers.xvlm_util.utils import get_config

#     config, model_path = get_config("xvlm-coco")

#     model = XVLM(config)

#     model.load_pretrained(
#         model_path,
#         config,
#         is_eval=True,
#         is_pretrained=False,  # never used pretrained in NegCLIP paper?
#     )

#     if config["use_roberta"]:
#         tokenizer = RobertaTokenizer.from_pretrained(config["text_encoder"])
#     else:
#         # TODO: Hack. We should use the tokenizer from the config
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     return XVLMModel(
#         model=model,
#         model_name=model_name,
#         tokenizer=tokenizer,
#         norm_mean=OPENAI_CLIP_MEAN,
#         norm_std=OPENAI_CLIP_STD,
#         input_resolution=384,
#         **kwargs
#     ), [
#         "zeroshot_classification",
#         "zeroshot_relation",
#     ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 32",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitB32_400m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 32",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitB32_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1011,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT g 14",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitG14_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-g-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1843,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT G 14",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitbigG14_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 16",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitB16_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 16",
        "year": 2021,
        "month": 11,
    },
)
def openclip_vitB16_400m(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "OpenCOCA ViT L 14",
        "year": 2022,
        "month": 5,
    },
)
def opencoca_vitL14_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "coca_ViT-L-14", pretrained="laion2b_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=76,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "OpenCOCA ViT B 32",
        "year": 2022,
        "month": 5,
    },
)
def opencoca_vitB32_2b(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "coca_ViT-B-32", pretrained="laion2b_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("coca_ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=76,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Negative CLIP",
        "architecture": "vit",
        "name": "NegCLIP ViT B 32",
        "year": 2023,
        "month": 3,
    },
)
def negclip_vitB32(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    path = os.path.join(HUB_CACHE_DIR, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegCLIP model...")
        import gdown

        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=path, load_weights_only=False
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT B 16",
        "year": 2021,
        "month": 1,
    },
)
def clip_vitB16(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("ViT-B/16", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet50(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("RN50", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet50_quickgelu(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
        "year": 2021,
        "month": 7,
    },
)
def clip_resnet50_quickgelu_yfcc15m(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="yfcc15m"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 12,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
        "year": 2021,
        "month": 7,
    },
)
def clip_resnet50_quickgelu_cc12m(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="cc12m"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet101",
        "year": 2021,
        "month": 7,
    },
)
def openclip_resnet101_yfcc(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("RN101", pretrained="yfcc15m")

    tokenizer = open_clip.get_tokenizer("RN101")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet50",
        "year": 2021,
        "month": 7,
    },
)
def openclip_resnet50_yfcc(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="yfcc15m")

    tokenizer = open_clip.get_tokenizer("RN50")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 12,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet50",
        "year": 2021,
        "month": 7,
    },
)
def openclip_resnet50_cc(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="cc12m")

    tokenizer = open_clip.get_tokenizer("RN50")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet101(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("RN101", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101 GeLU",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet101_quickgelu(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "RN101-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("RN101-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101 GeLU",
        "year": 2021,
        "month": 7,
    },
)
def clip_resnet101_quickgelu_yfcc15m(model_name, **kwargs):
    import open_clip
    from unibench.models_zoo.wrappers import ClipModel

    model, _, _ = open_clip.create_model_and_transforms(
        "RN101-quickgelu", pretrained="yfcc15m"
    )

    tokenizer = open_clip.get_tokenizer("RN101-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 87,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x4",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet50x4(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("RN50x4", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 167,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x16",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet50x16(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("RN50x16", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 420,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x64",
        "year": 2021,
        "month": 1,
    },
)
def clip_resnet50x64(model_name, **kwargs):
    from unibench.models_zoo.wrappers import ClipModel
    import clip

    model, _ = clip.load("RN50x64", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    ), [
        "zeroshot_classification",
        "zeroshot_relation",
    ]
