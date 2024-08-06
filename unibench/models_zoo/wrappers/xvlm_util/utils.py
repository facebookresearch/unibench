"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import subprocess
import yaml

from unibench.common_utils import HUB_CACHE_DIR
import gdown

download_urls = {
    "xvlm-flickr": {
        "model_url": "1vhdtH3iFaoZuMqOGm-8YM-diPWVfRJzv",
        "vision_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_swinB_384.json",
        "config_url": "13-GCckeAh7QUeFVGwye7qLJamwl_hXdf",
        "bert_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_bert.json",
    },
    "xvlm-coco": {
        "model_url": "1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN",
        "vision_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_swinB_384.json",
        "config_url": "11pdOukGXZzmPubvjLhJ2Sr1BIBRTEM-P",
        "bert_config_url": "https://github.com/zengyan-97/X-VLM/raw/e7b960256d194952321b5adad39770c03e6ce9c2/configs/config_bert.json",
    },
}


def get_config(version, root_dir=HUB_CACHE_DIR):
    config_path = os.path.join(root_dir, f"{version}-config")
    model_path = os.path.join(root_dir, f"{version}.pth")
    bert_config_path = os.path.join(
        root_dir,
        "configs",
        download_urls[version]["bert_config_url"].split("/")[-1],
    )
    vision_config_path = os.path.join(
        root_dir,
        "configs",
        download_urls[version]["vision_config_url"].split("/")[-1],
    )

    if not (
        os.path.exists(config_path)
        and os.path.exists(model_path)
        and os.path.exists(bert_config_path)
        and os.path.exists(vision_config_path)
    ):
        print(f"Downloading XVLM model to {root_dir}...")
        model_url = download_urls[version]["model_url"]
        config_url = download_urls[version]["config_url"]
        bert_config_url = download_urls[version]["bert_config_url"]
        vision_config_url = download_urls[version]["vision_config_url"]
        os.makedirs(os.path.join(root_dir, "configs"), exist_ok=True)
        gdown.download(id=model_url, output=model_path, quiet=False)
        gdown.download(id=config_url, output=config_path, quiet=False)
        subprocess.call(["wget", "-c", bert_config_url, "-O", bert_config_path])
        subprocess.call(["wget", "-c", vision_config_url, "-O", vision_config_path])

    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    config["vision_config"] = vision_config_path
    config["text_config"] = bert_config_path

    return config, model_path
