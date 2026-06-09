from functools import partial

import fire

from unibench import Evaluator
from unibench.benchmarks_zoo.handlers.clip_handlers import ZeroShotBenchmarkHandler
from unibench.benchmarks_zoo.handlers.vllm_handlers import MultiChoiceClassificationBenchmarkHandler
from unibench.benchmarks_zoo.wrappers.huggingface import HuggingFaceDataset
from unibench.common_utils.constants import OUTPUT_DIR

# ── Model list ────────────────────────────────────────────────────────────────
# API-based models only — no GPU required.
MODELS = [
    "gpt_4o",
    "gpt_4o_mini",
    "gpt_4_1",
    "gpt_5_4_genai_responses",
    "claude_4_6_opus_genai_vertex",
    "gemini_3_1_pro_preview_fair",
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
        kwargs["model_id"] = int(idx)

    return Evaluator(**kwargs)


def _run_relation(evaluator):
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_classification", "multi_choice_relation"],
    )


def _run_vqa(evaluator):
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_vqa"],
    )


def _run_classification(evaluator, benchmark_id, small_set):
    benchmark_name = CLASSIFICATION_BENCHMARKS[benchmark_id]
    download_name = BENCHMARK_NAME_MAPPING.get(benchmark_name, benchmark_name)

    meta = HuggingFaceDataset(transform=None, dataset_url=f"haideraltahan/wds_{download_name}")
    class_names = meta.classes
    templates = meta.templates

    if small_set:
        num_classes_list = [c for c in [32] if c <= len(class_names)] or [len(class_names)]
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
        tasks=["multi_choice_classification"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(
    output_dir=OUTPUT_DIR,
    num_workers=4,
    idx=0,
    model_name=None,
    mode="vqa",
    benchmark_id=0,
    small_set=True,
):
    """Evaluation script for API-based models (no GPU required).

    Args:
        output_dir:   Directory to write results.
        num_workers:  DataLoader workers.
        idx:          Model index into MODELS list, or 'all' to run every vllm model.
        model_name:   Explicit model name; overrides idx when provided.
        mode:         One of 'relation', 'classification', 'vqa', or 'all'.
        benchmark_id: Index into CLASSIFICATION_BENCHMARKS (0-44). Used when mode
                      includes 'classification'. Pass $SLURM_ARRAY_TASK_ID.
        small_set:    If True, evaluates with only 32 class choices (faster).
    """
    if mode not in ("relation", "classification", "vqa", "all"):
        raise ValueError(f"mode must be 'relation', 'classification', 'vqa', or 'all'; got '{mode}'")

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
