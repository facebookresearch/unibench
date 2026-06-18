from functools import partial

import fire

from unibench import Evaluator
from unibench.benchmarks_zoo.handlers.clip_handlers import ZeroShotBenchmarkHandler
from unibench.benchmarks_zoo.handlers.vllm_handlers import MultiChoiceClassificationBenchmarkHandler
from unibench.benchmarks_zoo.wrappers.huggingface import HuggingFaceDataset
from unibench.common_utils.constants import OUTPUT_DIR

# ── Model list ────────────────────────────────────────────────────────────────
MODELS = [
    "qwen_3_5_0_8b",
    "qwen_3_5_2b",
    "qwen_3_5_4b",
    "qwen_3_5_9b",
    "qwen_3_5_27b",
    "qwen_3_5_122b_a10b",
    "kimi_k2_5",
    "qwen_3_5_35b_a3b",
    "glm_4_5",
    "qwen_3_2b",
    "qwen_3_32b",
    "qwen_3_8b",
    "qwen_3_4b",
    "siglip2_so400_14_378",
    "qwen_2_5_3b",
    "siglip_so400_14",
    "clip_vitL14",
    "eva01_vitG14_plus_2b",
    "llava_1_5_13b",
    "llava_1_5_7b",
    "llava_1_6_mistral_7b",
    "llava_1_6_vicuna_13b",
    "llava_1_6_vicuna_7b",
    "llava_next_llama_8b",
    "aya_vision_32b",
    "aya_vision_8b",
    "gemma3_4b",
    "gemma3_12b",
    "paligemma_3b_mix_224",
    "paligemma2_3b_mix_224",
    "paligemma2_10b_mix_224",
    "paligemma2_10b_mix_448",
    "paligemma2_28b_mix_224",
    "paligemma2_3b_mix_448",
    "paligemma_3b_mix_448",
]

# ── Classification benchmarks (indices 0-44) ──────────────────────────────────
CLASSIFICATION_BENCHMARKS = [
    "caltech101",
    "cars",
    "cifar10",
    "cifar100",
    "clevr_count",
    "clevr_distance",
    "country211",
    "cub",
    "dmlab",
    "dspr_orientation",
    "dspr_x_position",
    "dspr_y_position",
    "dtd",
    "eurosat",
    "fashion_mnist",
    "fgvc_aircraft",
    "flowers102",
    "food101",
    "gtsrb",
    "imagenet1k",
    "imagenet9",
    "imagenet_sketch",
    "imageneta",
    "imagenetc",
    "imagenete",
    "imageneto",
    "imagenetr",
    "imagenetv2",
    "inaturalist",
    "kitti_distance",
    "mnist",
    "objectnet",
    "pcam",
    "pets",
    "places365",
    "pug_imagenet",
    "renderedsst2",
    "resisc45",
    "retinopathy",
    "smallnorb_azimuth",
    "smallnorb_elevation",
    "stl10",
    "sun397",
    "svhn",
    "voc2007",
]

BENCHMARK_NAME_MAPPING = {
    "clevr_distance": "clevr_closest_object_distance",
    "clevr_count": "clevr_count_all",
    "smallnorb_elevation": "smallnorb_label_elevation",
    "smallnorb_azimuth": "smallnorb_label_azimuth",
    "retinopathy": "diabetic_retinopathy",
    "kitti_distance": "kitti_closest_vehicle_distance",
    "flowers102": "flowers",
    "dspr_y_position": "dsprites_label_y_position",
    "dspr_x_position": "dsprites_label_x_position",
    "dspr_orientation": "dsprites_label_orientation",
}

# ── Relation / composition benchmarks ─────────────────────────────────────────
RELATION_BENCHMARKS = [
    "countbench",
    "vg_relation",
    "flickr30k_order",
    "sugarcrepe",
    "bivlc",
    "winoground",
    "vg_attribution",
    "coco_order",
]

# ── VQA benchmarks ────────────────────────────────────────────────────────────
VQA_BENCHMARKS = [
    "openapps",
    "mmmu_pro",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_evaluator(output_dir, num_workers, idx, model_name, benchmarks=None):
    """Return a configured Evaluator instance."""
    kwargs = dict(
        download_aggregate_precomputed=False,
        num_workers=num_workers,
        output_dir=output_dir,
    )
    if benchmarks is not None:
        kwargs["benchmarks"] = benchmarks

    if model_name is not None:
        kwargs["models"] = [model_name]
    elif idx == "all":
        kwargs["models"] = "vllm"
    else:
        kwargs["models"] = MODELS
        kwargs["model_id"] = idx

    return Evaluator(**kwargs)


def _run_relation(evaluator):
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_classification", "multi_choice_relation", "zeroshot_relation"],
    )


def _run_vqa(evaluator):
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_vqa", "vqa_multiple_choice"],
    )


def _run_classification(evaluator, benchmark_id, small_set):
    benchmark_name = CLASSIFICATION_BENCHMARKS[benchmark_id]
    download_name = BENCHMARK_NAME_MAPPING.get(benchmark_name, benchmark_name)

    # Load dataset metadata only (no transform needed)
    meta = HuggingFaceDataset(transform=None, dataset_url=f"haideraltahan/wds_{download_name}")
    class_names = meta.classes
    templates = meta.templates

    # Determine which class-count variants to run
    if small_set:
        num_classes_list = [c for c in [10] if c <= len(class_names)] or [len(class_names)]
    else:
        num_classes_list = [2 ** i for i in range(1, 40) if 2 ** i <= len(class_names)]
        if not num_classes_list or num_classes_list[-1] < len(class_names):
            num_classes_list.append(len(class_names))

    benchmark_factory = partial(
        HuggingFaceDataset,
        dataset_url=f"haideraltahan/wds_{download_name}",
        max_num_samples=5000,
    )

    bench_names = []
    for num in num_classes_list:
        name = f"{benchmark_name}_{num}"
        print(f"Adding benchmark: {name} with {num} classes")
        evaluator.add_benchmark(
            benchmark_name=name,
            benchmark=benchmark_factory,
            handlers={
                "multi_choice_classification": partial(
                    MultiChoiceClassificationBenchmarkHandler,
                    class_names=class_names,
                    num_classes=num,
                ),
                "zeroshot_classification": partial(
                    ZeroShotBenchmarkHandler,
                    class_names=class_names,
                    templates=templates,
                    num_classes=num,
                ),
            },
            meta_data={"benchmark_type": "object recognition"},
        )
        bench_names.append(name)

    evaluator.update_benchmark_list(bench_names)
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_classification", "zeroshot_classification"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(
    output_dir=OUTPUT_DIR,
    num_workers=8,
    idx=1,
    model_name=None,
    mode="relation",
    benchmark_id=0,
    small_set=True,
):
    """Combined evaluation script.

    Args:
        output_dir:   Directory to write results.
        num_workers:  DataLoader workers per GPU.
        idx:          Model index into MODELS list, or 'all' to run every vllm model.
        model_name:   Explicit model name; overrides idx when provided.
        mode:         One of 'relation', 'classification', 'vqa', or 'all'.
                      'relation'       – runs relation/composition benchmarks (main.py behaviour)
                      'classification' – runs classification benchmarks  (imagenet_main.py behaviour)
                      'vqa'            – runs VQA benchmarks (openapps, mmmu_pro)
                      'all'            – runs all three
        benchmark_id: Index into CLASSIFICATION_BENCHMARKS (0-44). Used when mode includes
                      'classification'. Pass $SLURM_ARRAY_TASK_ID from the job script.
        small_set:    If True, evaluates with only 32 class choices (faster).
                      If False, evaluates across all powers-of-2 up to the full class count.
    """
    if mode not in ("relation", "classification", "vqa", "all"):
        raise ValueError(f"mode must be 'relation', 'classification', 'vqa', or 'all'; got '{mode}'")

    # For classification mode benchmarks are added dynamically; for relation/vqa
    # the benchmark list is passed to the Evaluator constructor.
    if mode == "classification":
        evaluator = _build_evaluator(output_dir, num_workers, idx, model_name, benchmarks=None)
    elif mode == "relation":
        evaluator = _build_evaluator(
            output_dir, num_workers, idx, model_name, benchmarks=RELATION_BENCHMARKS
        )
    elif mode == "vqa":
        evaluator = _build_evaluator(
            output_dir, num_workers, idx, model_name, benchmarks=VQA_BENCHMARKS
        )
    else:  # all
        evaluator = _build_evaluator(
            output_dir, num_workers, idx, model_name,
            benchmarks=RELATION_BENCHMARKS + VQA_BENCHMARKS,
        )

    if mode in ("relation", "all"):
        _run_relation(evaluator)

    if mode in ("vqa", "all"):
        _run_vqa(evaluator)

    if mode in ("classification", "all"):
        _run_classification(evaluator, benchmark_id=int(benchmark_id), small_set=small_set)


if __name__ == "__main__":
    fire.Fire(main)
