from functools import partial
from typing import List

import fire
import torch
from unibench import Evaluator
from unibench.benchmarks_zoo.benchmarks import imagenet1k
from unibench.benchmarks_zoo.handlers import ZeroShotBenchmarkHandler
from unibench.benchmarks_zoo.wrappers import HuggingFaceDataset
import requests
import json
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from unibench.benchmarks_zoo.handlers.vllm_handlers import TextClassificationBenchmarkHandler


def main(model_id: int = 2, num_workers: int = 8):
    # Create benchmark using TestDataset with the correct paths
    benchmark = HuggingFaceDataset(
        transform=None, dataset_url="haideraltahan/wds_imagenet1k"
    )

    # Get class names from the benchmark dataset
    class_names = benchmark.classes

    benchmark = partial(
        HuggingFaceDataset,
        dataset_url="haideraltahan/wds_imagenet1k"
    )

    eval = Evaluator(
        model_id=model_id,
        num_workers=num_workers,
        models=[
        'llava_1_5_7b',
        'llava_next_llama_8b',
        'chameleon_7b',
        'paligemma_3b_224',
        'paligemma_3b_mix_224'
    ])

    eval.add_benchmark(
        benchmark_name="imagenet1k_1000",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=1000,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_2",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=2,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_4",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=4,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_8",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=8,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_16",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=16,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_32",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=32,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_64",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=64,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_128",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=128,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_256",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=256,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.add_benchmark(
        benchmark_name="imagenet1k_512",
        benchmark=benchmark,
        handlers={
            "text_classification": partial(
                TextClassificationBenchmarkHandler,
                class_names=class_names,
                num_classes=512,
            ),
        },
        meta_data={
            "benchmark_type": "object recognition",
        },
    )
    eval.update_benchmark_list(["imagenet1k_1000", "imagenet1k_2", "imagenet1k_4", "imagenet1k_8", "imagenet1k_16", "imagenet1k_32", "imagenet1k_64", "imagenet1k_128", "imagenet1k_256", "imagenet1k_512"])
    eval.evaluate()

if __name__ == "__main__":
    fire.Fire(main)
