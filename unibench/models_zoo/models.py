"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from unibench.models_zoo import register_model
from unibench.common_utils.constants import HUB_CACHE_DIR, CURRENT_DIR
from unibench.models_zoo.wrappers import *
import timm
import open_clip
import clip
from git import Repo
import sys


def load_blip(model_name, model_url, model_size="base", image_size=224, **kwargs):
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
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        use_itm_head=True,
        input_resolution=image_size,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 14,
        "model_size": 86,
        "learning_objective": "BLIP",
        "architecture": "vit",
        "name": "BLIP ViT B 16 ",
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
    },
)
def eva02_vitE14_plus_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-E-14-plus", pretrained="laion2b_s9b_b144k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
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
    },
)
def eva02_vitE14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-E-14", pretrained="laion2b_s4b_b115k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-E-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT L 14",
    },
)
def eva02_vitL14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-L-14", pretrained="merged2b_s4b_b131k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "EVA02",
        "architecture": "vit",
        "name": "EVA02 ViT B 16",
    },
)
def eva02_vitB16_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-B-16", pretrained="merged2b_s8b_b131k"
    )

    tokenizer = open_clip.get_tokenizer("EVA02-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1011,
        "learning_objective": "EVA01",
        "architecture": "vit",
        "name": "EVA01 ViT g 14",
    },
)
def eva01_vitG14_plus_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA01-g-14-plus", pretrained="merged2b_s11b_b114k"
    )

    tokenizer = open_clip.get_tokenizer("EVA01-g-14-plus")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 1011,
        "learning_objective": "EVA01",
        "architecture": "vit",
        "name": "EVA01 ViT g 14",
    },
)
def eva01_vitG14_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA01-g-14", pretrained="laion400m_s11b_b41k"
    )

    tokenizer = open_clip.get_tokenizer("EVA01-g-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 1843,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT G 14",
    },
)
def clipa_vitbigG14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 22,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-S",
    },
)
def vitamin_s_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 22,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-S-LTT",
    },
)
def vitamin_s_ltt_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 87,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-B",
    },
)
def vitamin_b_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 87,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-B-LTT",
    },
)
def vitamin_b_ltt_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L",
    },
)
def vitamin_l_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2",
    },
)
def vitamin_l2_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2-256",
    },
)
def vitamin_l2_256_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L2-336",
    },
)
def vitamin_l2_336_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-256",
    },
)
def vitamin_xl_256_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-336",
    },
)
def vitamin_xl_336_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 436,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-XL-384",
    },
)
def vitamin_xl_384_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L-256",
    },
)
def vitamin_l_256_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 333,
        "learning_objective": "ViTamin",
        "architecture": "vitamin",
        "name": "ViTamin-L-336",
    },
)
def vitamin_l_336_1b(model_name, **kwargs):
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
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 633,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT H 14",
    },
)
def clipa_vitH14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 307,
        "learning_objective": "CLIPA",
        "architecture": "vit",
        "name": "CLIPA ViT L 14",
    },
)
def clipa_vitL14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-CLIPA", pretrained="datacomp1b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-CLIPA")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=32,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT L 16",
    },
)
def siglip_vitL16(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "Roberta ViT B 32",
    },
)
def roberta_vitB32(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "roberta-ViT-B-32", pretrained="laion2b_s12b_b32k"
    )

    tokenizer = open_clip.get_tokenizer("roberta-ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )

@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16",
    },
)
def siglip_vitB16(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 256",
    },
)
def siglip_vitB16_256(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 384",
    },
)
def siglip_vitB16_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT B 16 512",
    },
)
def siglip_vitB16_512(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP ViT L 16 384",
    },
)
def siglip_vitL16_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14",
    },
)
def siglip_so400_14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14 378",
    },
)
def siglip_so400_14_378(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-378", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-378")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP So400 14 384",
    },
)
def siglip_so400_14_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=16,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 512",
    },
)
def siglip2_so400_16_512(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 512",
    },
)
def siglip2_so400_16_512(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 384",
    },
)
def siglip2_so400_16_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 16 256",
    },
)
def siglip2_so400_16_256(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 14 378",
    },
)
def siglip2_so400_14_378(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP2-378", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2-378")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 400,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "So400",
        "name": "SigLIP 2 So400 14",
    },
)
def siglip2_so400_14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP2", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP2")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 512",
    },
)
def siglip2_vitL16_512(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 384",
    },
)
def siglip2_vitL16_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 307,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT L 16 256",
    },
)
def siglip2_vitL16_256(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 512",
    },
)
def siglip2_vitB16_512(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-512", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-512")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 384",
    },
)
def siglip2_vitB16_384(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-384", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-384")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16 256",
    },
)
def siglip2_vitB16_256(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 16",
    },
)
def siglip2_vitB16(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 10000,
        "model_size": 86,
        "learning_objective": "Contrastive (sigmoid-based)",
        "architecture": "vit",
        "name": "SigLIP 2 ViT B 32 256",
    },
)
def siglip2_vitB32_256(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-SigLIP2-256", pretrained="webli"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-SigLIP2-256")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.IMAGENET_INCEPTION_MEAN,
        norm_std=timm.data.constants.IMAGENET_INCEPTION_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=64,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 32",
    },
)
def openclip_vitB32_metaclip_fullcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 16",
    },
)
def openclip_vitB16_metaclip_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 32",
    },
)
def openclip_vitB32_metaclip_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "ViT B 32 GeLU",
    },
)
def openclip_vitB32_quickgelu_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "ViT B 32 GeLU",
    },
)
def openclip_vitB32_quickgelu_openai(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT B 16",
    },
)
def openclip_vitB16_metaclip_fullcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
    },
)
def openclip_vitL14_dfn2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="dfn2b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT L 14",
    },
)
def openclip_vitL14_metaclip_400(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="metaclip_400m"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT L 14",
    },
)
def openclip_vitL14_metaclip_fullcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2500,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "MetaCLIP ViT H 14",
    },
)
def openclip_vitH14_metaclip_fullcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="metaclip_fullcc"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 5000,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT H 14",
    },
)
def openclip_vitH14_dfn5b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="dfn5b"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 88,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ConvNext",
    },
)
def openclip_convnext_base(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_base", pretrained="laion400m_s13b_b51k"
    )

    tokenizer = open_clip.get_tokenizer("convnext_base")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "clipHero",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT B 32",
    },
)
def clip_vitB32(model_name, **kwargs):
    model, _ = clip.load("ViT-B/32", download_root=str(HUB_CACHE_DIR))

    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 13,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
    },
)
def openclip_vitB32_datacomp_s(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_s_s13m_b4k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 128,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
    },
)
def openclip_vitB32_datacomp_m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_m_s128m_b4k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 32",
    },
)
def openclip_vitB32_datacomp_xl(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 16",
    },
)
def openclip_vitB16_datacomp_xl(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 1280,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT B 16",
    },
)
def openclip_vitB16_datacomp_l(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="datacomp_l_s1b_b8k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 633,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT H 14",
    },
)
def openclip_vitH14(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 16,
        "model_size": 86,
        "learning_objective": "XVLM",
        "architecture": "Swin",
        "name": "XVLM Swin B",
    },
)
def xvlm_flickr(model_name, **kwargs):
    from .wrappers.xvlm_util.xvlm import XVLM
    from .wrappers.xvlm_util.tokenization_bert import BertTokenizer
    from .wrappers.xvlm_util.tokenization_roberta import RobertaTokenizer
    from .wrappers.xvlm_util.utils import get_config

    config, model_path = get_config("xvlm-flickr")

    model = XVLM(config)

    model.load_pretrained(
        model_path,
        config,
        is_eval=True,
        is_pretrained=False,  # never used pretrained in NegCLIP paper?
    )

    if config["use_roberta"]:
        tokenizer = RobertaTokenizer.from_pretrained(config["text_encoder"])
    else:
        # TODO: Hack. We should use the tokenizer from the config
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return XVLMModel(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 70,
        "model_size": 86,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "FLAVA ViT B 32",
    },
)
def flava_full(model_name, **kwargs):
    from transformers import FlavaForPreTraining, FlavaImageProcessor, BertTokenizer

    model = FlavaForPreTraining.from_pretrained("facebook/flava-full")

    processor = FlavaImageProcessor.from_pretrained("facebook/flava-full")

    tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

    return FlavaModel(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=processor.size["height"],
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
    },
)
def openclip_vitL14_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 12800,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "DataComp ViT L 14",
    },
)
def openclip_vitL14_datacomp_xl(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT L 14",
    },
)
def openclip_vitL14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 307,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT L 14",
    },
)
def clip_vitL14(model_name, **kwargs):
    model, _ = clip.load("ViT-L/14", download_root=str(HUB_CACHE_DIR))

    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 16,
        "model_size": 86,
        "learning_objective": "XVLM",
        "architecture": "Swin",
        "name": "XVLM Swin B",
    },
)
def xvlm_coco(model_name, **kwargs):
    from .wrappers.xvlm_util.xvlm import XVLM
    from .wrappers.xvlm_util.tokenization_bert import BertTokenizer
    from .wrappers.xvlm_util.tokenization_roberta import RobertaTokenizer
    from .wrappers.xvlm_util.utils import get_config

    config, model_path = get_config("xvlm-coco")

    model = XVLM(config)

    model.load_pretrained(
        model_path,
        config,
        is_eval=True,
        is_pretrained=False,  # never used pretrained in NegCLIP paper?
    )

    if config["use_roberta"]:
        tokenizer = RobertaTokenizer.from_pretrained(config["text_encoder"])
    else:
        # TODO: Hack. We should use the tokenizer from the config
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return XVLMModel(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=384,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 32",
    },
)
def openclip_vitB32_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 32",
    },
)
def openclip_vitB32_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1011,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT g 14",
    },
)
def openclip_vitG14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-g-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 1843,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT G 14",
    },
)
def openclip_vitbigG14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 16",
    },
)
def openclip_vitB16_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "OpenCLIP ViT B 16",
    },
)
def openclip_vitB16_400m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion400m_e32"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 307,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "OpenCOCA ViT L 14",
    },
)
def opencoca_vitL14_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "coca_ViT-L-14", pretrained="laion2b_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=76,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 2000,
        "model_size": 86,
        "learning_objective": "Other",
        "architecture": "vit",
        "name": "OpenCOCA ViT B 32",
    },
)
def opencoca_vitB32_2b(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "coca_ViT-B-32", pretrained="laion2b_s13b_b90k"
    )

    tokenizer = open_clip.get_tokenizer("coca_ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        context_length=76,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Negative CLIP",
        "architecture": "vit",
        "name": "NegCLIP ViT B 32",
    },
)
def negclip_vitB32(model_name, **kwargs):
    path = os.path.join(HUB_CACHE_DIR, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegCLIP model...")
        import gdown

        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=path, load_weights_only=False)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size[0],
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 86,
        "learning_objective": "Contrastive",
        "architecture": "vit",
        "name": "CLIP ViT B 16",
    },
)
def clip_vitB16(model_name, **kwargs):
    model, _ = clip.load("ViT-B/16", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50",
    },
)
def clip_resnet50(model_name, **kwargs):
    model, _ = clip.load("RN50", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
    },
)
def clip_resnet50_quickgelu(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
    },
)
def clip_resnet50_quickgelu_yfcc15m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="yfcc15m"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 12,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50 GeLU",
    },
)
def clip_resnet50_quickgelu_cc12m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="cc12m"
    )

    tokenizer = open_clip.get_tokenizer("RN50-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet101",
    },
)
def openclip_resnet101_yfcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms("RN101", pretrained="yfcc15m")

    tokenizer = open_clip.get_tokenizer("RN101")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet50",
    },
)
def openclip_resnet50_yfcc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="yfcc15m")

    tokenizer = open_clip.get_tokenizer("RN50")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 12,
        "model_size": 38,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "OpenCLIP ResNet50",
    },
)
def openclip_resnet50_cc(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="cc12m")

    tokenizer = open_clip.get_tokenizer("RN50")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101",
    },
)
def clip_resnet101(model_name, **kwargs):
    model, _ = clip.load("RN101", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101 GeLU",
    },
)
def clip_resnet101_quickgelu(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "RN101-quickgelu", pretrained="openai"
    )

    tokenizer = open_clip.get_tokenizer("RN101-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 15,
        "model_size": 56,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet101 GeLU",
    },
)
def clip_resnet101_quickgelu_yfcc15m(model_name, **kwargs):
    model, _, _ = open_clip.create_model_and_transforms(
        "RN101-quickgelu", pretrained="yfcc15m"
    )

    tokenizer = open_clip.get_tokenizer("RN101-quickgelu")

    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.image_size,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 87,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x4",
    },
)
def clip_resnet50x4(model_name, **kwargs):
    model, _ = clip.load("RN50x4", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 167,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x16",
    },
)
def clip_resnet50x16(model_name, **kwargs):
    model, _ = clip.load("RN50x16", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )


@register_model(
    "vision_text",
    {
        "dataset_size": 400,
        "model_size": 420,
        "learning_objective": "Contrastive",
        "architecture": "conv",
        "name": "CLIP ResNet50x64",
    },
)
def clip_resnet50x64(model_name, **kwargs):
    model, _ = clip.load("RN50x64", download_root=str(HUB_CACHE_DIR))
    tokenizer = clip.tokenize
    return ClipModel(
        model,
        model_name,
        tokenizer=tokenizer,
        norm_mean=timm.data.constants.OPENAI_CLIP_MEAN,
        norm_std=timm.data.constants.OPENAI_CLIP_STD,
        input_resolution=model.visual.input_resolution,
        logit_scale=model.logit_scale,
        **kwargs
    )
