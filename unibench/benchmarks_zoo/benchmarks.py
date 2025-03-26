"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from unibench.benchmarks_zoo import register_benchmark
from .wrappers import *


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [32, 32],
        "num_classes": 10,
        "llama2_ppi": 211872.54,
    },
)
def cifar10(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_cifar10", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "imagenet",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "imagenet",
        "curated": False,
        "object_centric": True,
        "image_resolution": [490.38, 430.25],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenet1k(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenet1k", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [32, 32],
        "num_classes": 100,
        "llama2_ppi": 330736.54,
    },
)
def cifar100(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_cifar100", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [495.81, 475.08],
        "num_classes": 101,
        "llama2_ppi": 810.82,
    },
)
def food101(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_food101", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [701.17, 483.74],
        "num_classes": 196,
        "llama2_ppi": 15.09,
    },
)
def cars(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_cars", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [1098.57, 746.99],
        "num_classes": 100,
        "llama2_ppi": 385.1,
    },
)
def fgvc_aircraft(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_fgvc_aircraft", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "specifies classification",
        "curated": False,
        "object_centric": True,
        "image_resolution": [443.46, 399.37],
        "num_classes": 37,
        "llama2_ppi": 655.6,
    },
)
def pets(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_pets", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "texture",
        "capability": "texture detection",
        "curated": True,
        "object_centric": False,
        "image_resolution": [488.97, 447.49],
        "num_classes": 47,
        "llama2_ppi": 22521.58,
    },
)
def dtd(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_dtd", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "scene recognition",
        "curated": False,
        "object_centric": False,
        "image_resolution": [958.89, 775.09],
        "num_classes": 397,
        "llama2_ppi": 117271.23,
    },
)
def sun397(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_sun397", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [311.56, 241.55],
        "num_classes": 102,
        "llama2_ppi": 124394.14,
    },
)
def caltech101(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_caltech101", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "character recognition",
        "curated": True,
        "object_centric": True,
        "image_resolution": [28, 28],
        "num_classes": 10,
        "llama2_ppi": 36.87,
    },
)
def mnist(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


# @register_dataset(
#     "mnist",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_top2(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=benchmark.templates,
#         topx=2
# )

# @register_dataset(
#     "mnist",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_top3(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=benchmark.templates,
#         topx=3
#     )

# @register_dataset(
#     "mnist",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_top4(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=benchmark.templates,
#         topx=4
#     )

# @register_dataset(
#     "mnist",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_top5(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=benchmark.templates,
#         topx=5
#     )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "character recognition",
        "curated": True,
        "object_centric": True,
        "image_resolution": [28, 28],
        "num_classes": 10,
        "llama2_ppi": None,
    },
)
def fashion_mnist(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_fashion_mnist", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [28, 28],
        "num_classes": 151,
        "llama2_ppi": None,
    },
)
def pug_imagenet(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_pug_imagenet", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_more_prompts(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=[
#             "a photo of the number: '{}'.",
#             "a digit drawing of the number: '{}'.",
#             "a digit sketch of the number: '{}'.",
#             "a handwritten digit image of: '{}'.",
#             "a digit illustration of: '{}'.",
#             "a graphical representation of the number: '{}'.",
#             "a visual depiction of the digit: '{}'.",
#             "a snapshot of the numeral: '{}'.",
#             "a handwritten representation of the number: '{}'.",
#             "an image showcasing the digit: '{}'.",
#         ],
#     )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_numbers(benchmark_name, transform=None, **kwargs):
#     import inflect

#     p = inflect.engine()

#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=[p.number_to_words(i) for i in benchmark.classes],
#         templates=benchmark.templates,
#     )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_numbers_more_prompts(benchmark_name, transform=None, **kwargs):
#     import inflect

#     p = inflect.engine()

#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=[p.number_to_words(i) for i in benchmark.classes],
#         templates=[
#             "A photo of the number: '{}'.",
#             "A digit drawing of the number: '{}'.",
#             "A digit sketch of the number: '{}'.",
#             "A handwritten digit image of: '{}'.",
#             "A digit illustration of: '{}'.",
#             "A graphical representation of the number: '{}'.",
#             "A visual depiction of the digit: '{}'.",
#             "A snapshot of the numeral: '{}'.",
#             "A handwritten representation of the number: '{}'.",
#             "An image showcasing the digit: '{}'.",
#         ],
#     )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_numbers_diff_prompts(benchmark_name, transform=None, **kwargs):
#     import inflect

#     p = inflect.engine()

#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=[p.number_to_words(i) for i in benchmark.classes],
#         templates=[
#             "showcasing the digit {}, is this image.",
#             "this number {} is represented in a handwritten form.",
#             "the numeral {} is captured in this snapshot.",
#             "the digit {} is depicted visually in this image.",
#             "this image is a graphical representation of the number {}.",
#             "this is an illustration of the digit {}.",
#             "this image represents the digit {} in a handwritten form.",
#             "the number {} is sketched as a digit in this image.",
#             "this is a photograph of the digit {}.",
#             "the number {} is drawn as a digit in this image.",
#         ],
#     )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "object recognition",
#         "capability": "character recognition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [28, 28],
#         "num_classes": 10,
#         "llama2_ppi": 36.87,
#     },
# )
# def mnist_diff_prompts(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_mnist", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=[
#             "showcasing the digit {}, is this image.",
#             "this number {} is represented in a handwritten form.",
#             "the numeral {} is captured in this snapshot.",
#             "the digit {} is depicted visually in this image.",
#             "this image is a graphical representation of the number {}.",
#             "this is an illustration of the digit {}.",
#             "this image represents the digit {} in a handwritten form.",
#             "the number {} is sketched as a digit in this image.",
#             "this is a photograph of the digit {}.",
#             "the number {} is drawn as a digit in this image.",
#         ],
#     )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [50.50, 50.36],
        "num_classes": 43,
        "llama2_ppi": 4173.89,
    },
)
def gtsrb(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_gtsrb", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "character recognition",
        "curated": True,
        "object_centric": True,
        "image_resolution": [448, 448],
        "num_classes": 2,
        "llama2_ppi": 423531.1,
    },
)
def renderedsst2(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_renderedsst2", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [96, 96],
        "num_classes": 10,
        "llama2_ppi": 217277.01,
    },
)
def stl10(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_stl10", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "character recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [32, 32],
        "num_classes": 10,
        "llama2_ppi": 383003.08,
    },
)
def svhn(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_svhn", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "satellite",
        "curated": False,
        "object_centric": False,
        "image_resolution": [64, 64],
        "num_classes": 10,
        "llama2_ppi": 55325.21,
    },
)
def eurosat(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_eurosat", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "geographic diversity",
        "curated": False,
        "object_centric": False,
        "image_resolution": [468.59, 381.72],
        "num_classes": 211,
        "llama2_ppi": 1180.55,
    },
)
def country211(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_country211", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "rendition",
        "curated": True,
        "object_centric": True,
        "image_resolution": [762.64, 727.06],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenet_sketch(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenet_sketch", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


# @register_dataset(
#     "transfer",
#     {
#         "benchmark": "zero-shot",
#         "benchmark_type": "non-natural images",
#         "capability": "rendition",
#         "curated": True,
#         "object_centric": True,
#         "image_resolution": [762.64, 727.06],
#         "num_classes": 1000,
#         "llama2_ppi": 76188.02,
#     },
# )
# def fer2013(benchmark_name, transform=None, **kwargs):
#     benchmark = HuggingFaceDataset(
#         transform=transform, dataset_url="haideraltahan/wds_fer2013", **kwargs
#     )
#     return ZeroShotDatasetHandler(
#         benchmark_name=benchmark_name,
#         benchmark=benchmark,
#         classes=benchmark.classes,
#         templates=benchmark.templates,
#     )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "depth estimation",
        "curated": True,
        "object_centric": False,
        "image_resolution": [480, 320],
        "num_classes": 6,
        "llama2_ppi": 3616.24,
    },
)
def dmlab(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_dmlab", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "challenging imagenet",
        "curated": True,
        "object_centric": True,
        "image_resolution": [443.32, 427.16],
        "num_classes": 200,
        "llama2_ppi": 50415.99,
    },
)
def imageneta(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imageneta", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "challenging imagenet",
        "curated": True,
        "object_centric": True,
        "image_resolution": [408.65, 358.35],
        "num_classes": 200,
        "llama2_ppi": 87577.13,
    },
)
def imageneto(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imageneto", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "rendition",
        "curated": True,
        "object_centric": True,
        "image_resolution": [480.30, 459.18],
        "num_classes": 200,
        "llama2_ppi": 69055.32,
    },
)
def imagenetr(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenetr", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "medical",
        "curated": True,
        "object_centric": False,
        "image_resolution": [96, 96],
        "num_classes": 2,
        "llama2_ppi": 53.05,
    },
)
def pcam(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_pcam", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "natural transformations",
        "curated": True,
        "object_centric": True,
        "image_resolution": [296.13, 284.14],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenete(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenete", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "natural transformations",
        "curated": True,
        "object_centric": True,
        "image_resolution": [224, 224],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenet9(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenet9", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "corruption",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "corruption",
        "capability": "corruption",
        "curated": True,
        "object_centric": True,
        "image_resolution": [224, 224],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenetc(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenetc", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "specifies classification",
        "curated": False,
        "object_centric": True,
        "image_resolution": [632.38, 533.94],
        "num_classes": 102,
        "llama2_ppi": 10163.5,
    },
)
def flowers102(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_flowers", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "challenging imagenet",
        "curated": True,
        "object_centric": True,
        "image_resolution": [494.73, 417.16],
        "num_classes": 1000,
        "llama2_ppi": 76188.02,
    },
)
def imagenetv2(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_imagenetv2", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "robustness",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "robustness",
        "capability": "natural transformations",
        "curated": False,
        "object_centric": True,
        "image_resolution": [1238.76, 2265.51],
        "num_classes": 113,
        "llama2_ppi": 205745.07,
    },
)
def objectnet(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_objectnet", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "specifies classification",
        "curated": False,
        "object_centric": False,
        "image_resolution": [468.16, 385.79],
        "num_classes": 200,
        "llama2_ppi": 23365.88,
    },
)
def cub(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_cub", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "scene recognition",
        "curated": False,
        "object_centric": False,
        "image_resolution": [676.42, 551.72],
        "num_classes": 365,
        "llama2_ppi": 121139.79,
    },
)
def places365(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_places365", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "spatial understanding",
        "curated": True,
        "object_centric": False,
        "image_resolution": [480, 320],
        "num_classes": 6,
        "llama2_ppi": None,
    },
)
def clevr_distance(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_clevr_closest_object_distance",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "counting",
        "curated": True,
        "object_centric": False,
        "image_resolution": [480, 320],
        "num_classes": 8,
        "llama2_ppi": 410266.22,
    },
)
def clevr_count(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_clevr_count_all", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "counting",
        "curated": False,
        "object_centric": False,
        "image_resolution": None,
        "num_classes": 10,
        "llama2_ppi": None,
    },
)
def countbench(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_countbench", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "specifies classification",
        "curated": False,
        "object_centric": True,
        "image_resolution": [725.36, 664.10],
        "num_classes": 5089,
        "llama2_ppi": 845.3,
    },
)
def inaturalist(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_inaturalist", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "standard object recognition",
        "curated": False,
        "object_centric": True,
        "image_resolution": [144.95, 148.82],
        "num_classes": 20,
        "llama2_ppi": 594242.65,
    },
)
def voc2007(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_voc2007", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "satellite",
        "curated": False,
        "object_centric": False,
        "image_resolution": [256, 256],
        "num_classes": 45,
        "llama2_ppi": 323336.83,
    },
)
def resisc45(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_resisc45", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "pose detection",
        "curated": True,
        "object_centric": False,
        "image_resolution": [64, 64],
        "num_classes": 40,
        "llama2_ppi": 26.29,
    },
)
def dspr_orientation(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_dsprites_label_orientation",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "depth estimation",
        "curated": False,
        "object_centric": False,
        "image_resolution": [1238.87, 374.66],
        "num_classes": 4,
        "llama2_ppi": 124.44,
    },
)
def kitti_distance(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_kitti_closest_vehicle_distance",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "pose detection",
        "curated": True,
        "object_centric": False,
        "image_resolution": [96, 96],
        "num_classes": 18,
        "llama2_ppi": 28.15,
    },
)
def smallnorb_azimuth(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_smallnorb_label_azimuth",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "spatial understanding",
        "curated": True,
        "object_centric": False,
        "image_resolution": [96, 96],
        "num_classes": 9,
        "llama2_ppi": 35.36,
    },
)
def smallnorb_elevation(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_smallnorb_label_elevation",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "spatial understanding",
        "curated": True,
        "object_centric": False,
        "image_resolution": [64, 64],
        "num_classes": 32,
        "llama2_ppi": 24.8,
    },
)
def dspr_x_position(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_dsprites_label_x_position",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "reasoning",
        "capability": "spatial understanding",
        "curated": True,
        "object_centric": False,
        "image_resolution": [64, 64],
        "num_classes": 32,
        "llama2_ppi": 24.8,
    },
)
def dspr_y_position(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_dsprites_label_y_position",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "vtab",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "non-natural images",
        "capability": "medical",
        "curated": False,
        "object_centric": False,
        "image_resolution": [540, 511.83],
        "num_classes": 5,
        "llama2_ppi": 14.6,
    },
)
def retinopathy(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform,
        dataset_url="haideraltahan/wds_diabetic_retinopathy",
        **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "transfer",
    {
        "benchmark": "zero-shot",
        "benchmark_type": "object recognition",
        "capability": "geographic diversity",
        "curated": False,
        "object_centric": True,
        "image_resolution": [3381.92, 2669.03],
        "num_classes": 60,
        "llama2_ppi": 100158.6,
    },
)
def dollar_street(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_dollar_street", **kwargs
    )
    return ZeroShotBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
        classes=benchmark.classes,
        templates=benchmark.templates,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [400.53, 302.99],
        "num_classes": 2,
        "llama2_ppi": None,
    },
)
def vg_relation(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_vg_relation", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [457.79, 397.48],
        "num_classes": 5,
        "llama2_ppi": None,
    },
)
def flickr30k_order(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_flickr30k_order", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [569.52, 487.69],
        "num_classes": 2,
        "llama2_ppi": None,
    },
)
def sugarcrepe(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_sugarcrepe", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [569.52, 487.69],
        "num_classes": 2,
        "llama2_ppi": None,
    },
)
def bivlc(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_bivlc", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [1837.58, 1393.52],
        "num_classes": 2,
        "llama2_ppi": 691.08,
    },
)
def winoground(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_winoground", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [411.40, 326.25],
        "num_classes": 2,
        "llama2_ppi": None,
    },
)
def vg_attribution(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_vg_attribution", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )


@register_benchmark(
    "relation",
    {
        "benchmark": "relation",
        "benchmark_type": "relation",
        "capability": "relations",
        "curated": False,
        "object_centric": False,
        "image_resolution": [1837.58, 1393.52],
        "num_classes": 5,
        "llama2_ppi": None,
    },
)
def coco_order(benchmark_name, transform=None, **kwargs):
    benchmark = HuggingFaceDataset(
        transform=transform, dataset_url="haideraltahan/wds_coco_order", **kwargs
    )
    return RelationBenchmarkHandler(
        benchmark_name=benchmark_name,
        benchmark=benchmark,
    )
