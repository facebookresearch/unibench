"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

###### CREDIT GOES TO https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/datasets ######

from pathlib import Path
import fire
from tqdm import tqdm
import shutil
import webdataset
from class_names import CLASS_NAMES
from templates import TEMPLATES
from utils import *
import os
from benchmarks import SUN397
from datasets import load_dataset
import numpy as np

def main(
    dataset_name,
    root_dir="/research/haider/Datasets",
    language="en",
    upload2huggingface=True,
    image_format="webp",
    max_size=500_000_000,
    num_workers=64,
):
    transform = PIL_to_bytes(image_format)
    classnames = (
        CLASS_NAMES[language][dataset_name] if dataset_name in CLASS_NAMES[language] else None
    )
    templates = (
        TEMPLATES[language][dataset_name]
        if dataset_name in TEMPLATES[language]
        else TEMPLATES[language]["imagenet1k"]
    )

    if dataset_name == "sun397":
        ds = SUN397(root=root_dir, transform=transform, download=True, partition_idx=1)
        ds.templates = templates
    elif dataset_name == "bivlc":
        ds = load_dataset("imirandam/BiVLC", split="test", cache_dir=root_dir)
        ds = ds.map(lambda example: {"image": [transform(example["image"]), transform(example["negative_image"])], "captions": [example['caption'], example['negative_caption']], "split": f"{example['type']}\n{example['subtype']}"}, num_proc=num_workers)
        ds = [(example['image'], example['captions'], example['split']) for example in ds] 
        ds = ListDataset(ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    output_dir = Path(root_dir) / dataset_name
    split_dir = output_dir / "test"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    split_dir.mkdir(parents=True, exist_ok=True)
    
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0] # No collate, only for multiprocessing
    )
    
    
    if hasattr(ds, "classes") and ds.classes:
        classnames_fname = output_dir / "classnames.txt"
        with open(classnames_fname, "w") as classnames_file:
            print(*ds.classes, sep="\n", end="\n", file=classnames_file)
            print("Saved class names to '%s'" % classnames_fname)
    else:
        print("WARNING: No class names found")
        
    if hasattr(ds, "templates") and ds.templates:
        templates_fname = output_dir / "zeroshot_classification_templates.txt"
        with open(templates_fname, "w") as templates_file:
            print(*ds.templates, sep="\n", end="\n", file=templates_file)
            print("Saved class names to '%s'" % templates_fname)
    else:
        print("WARNING: No zeroshot classification templates found")

    data_fname = os.path.join(split_dir, r"%d.tar")
    sink = webdataset.ShardWriter(
        data_fname,
        maxsize=max_size
    )
    nsamples = 0
    for index, batch in enumerate(tqdm(dataloader, desc="Converting")):
        if len(batch) == 2:
            input, output = batch
        elif len(batch) == 3:
            input, output, split = batch
        else:
            raise ValueError(f"Unknown batch size: {len(batch)}")
            
        nsamples += 1
        
        if isinstance(input, bytes):
            input = {f"0.{image_format}": input}
        elif isinstance(input, list) or isinstance(input, torch.Tensor):
            input = {f"{i}.{image_format}": img for i, img in enumerate(input)}
        
        if isinstance(output, int):
            output = {'cls': output}
        elif isinstance(output, list):
            output = {'npy': np.array(output)}
            
        if split is not None:
            output['split.txt'] = split
        
        sink.write({
            "__key__": "s%07d" % index,
            **input,
            **output
        })
    num_shards = sink.shard
    sink.close()
    print("Saved dataset to '%s'" % data_fname.replace(r"%d", "{0..%d}" % (num_shards - 1)))
    nshards_fname = split_dir / "nshards.txt"
    with open(nshards_fname, "w") as nshards_file:
        print(num_shards, end="\n", file=nshards_file)
    print("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))
    print("Final dataset size:", nsamples)
    if upload2huggingface:
        os.system(f"huggingface-cli upload wds_{dataset_name} {str(output_dir)} {str(output_dir)} --repo-type dataset")
        

if __name__ == "__main__":
    fire.Fire(main)
