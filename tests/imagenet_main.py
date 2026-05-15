from functools import partial
from unibench.benchmarks_zoo.handlers.vllm_handlers import MultiChoiceClassificationBenchmarkHandler
from unibench.benchmarks_zoo.handlers.clip_handlers import ZeroShotBenchmarkHandler
from unibench import Evaluator
import fire

from unibench.benchmarks_zoo.wrappers.huggingface import HuggingFaceDataset
from unibench.common_utils.constants import OUTPUT_DIR
import inspect
import unibench.benchmarks_zoo as bz

def main(output_dir=OUTPUT_DIR, num_workers=8, idx=1, model_name=None, benchmark_id=36, small_set=True):

    if model_name is not None:
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models=[model_name],
            num_workers=num_workers,
            output_dir=output_dir,
        )
    elif idx == 'all':
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models="vllm",
            num_workers=num_workers,
            output_dir=output_dir,
        )
    else:
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models=[
                'qwen_3_5_0_8b',
                'qwen_3_5_2b',
                'qwen_3_5_4b',
                'qwen_3_5_9b',
                'qwen_3_5_27b',
                'qwen_3_5_122b_a10b',
                'kimi_k2_5',
                'qwen_3_5_35b_a3b',
                'glm_4_5',
                'qwen_3_2b',
                'qwen_3_32b',
                'qwen_3_8b', 
                'qwen_3_4b', 
                'siglip2_so400_14_378', 
                'qwen_2_5_3b', 
                'siglip_so400_14', 
                'clip_vitL14', 'eva01_vitG14_plus_2b', 'llava_1_5_13b', 'llava_1_5_7b', 
                 'llava_1_6_mistral_7b',
                 'llava_1_6_vicuna_13b', 'llava_1_6_vicuna_7b', 'llava_next_llama_8b', 'aya_vision_32b', 'aya_vision_8b', 'gemma3_4b', 'gemma3_12b',
                 'paligemma_3b_mix_224', 
                 'paligemma2_3b_mix_224', 
                 'paligemma2_10b_mix_224', 
                 'paligemma2_10b_mix_448', 
                 'paligemma2_28b_mix_224', 
                 'paligemma2_3b_mix_448', 
                 'paligemma_3b_mix_448'
                 ],
            model_id=idx,
            num_workers=num_workers,
            output_dir=output_dir,
        )

    benchmarks = ['caltech101', 'cars', 'cifar10', 'cifar100', 'clevr_count', 'clevr_distance', 'country211', 'cub', 'dmlab', 'dspr_orientation', 'dspr_x_position', 'dspr_y_position', 'dtd', 'eurosat', 'fashion_mnist', 'fgvc_aircraft', 'flowers102', 'food101', 'gtsrb', 'imagenet1k', 'imagenet9', 'imagenet_sketch', 'imageneta', 'imagenetc', 'imagenete', 'imageneto', 'imagenetr', 'imagenetv2', 'inaturalist', 'kitti_distance', 'mnist', 'objectnet', 'pcam', 'pets', 'places365', 'pug_imagenet', 'renderedsst2', 'resisc45', 'retinopathy', 'smallnorb_azimuth', 'smallnorb_elevation', 'stl10', 'sun397', 'svhn', 'voc2007']

    benchmark_name = benchmarks[benchmark_id]
    
    download_name = benchmark_name
    mapping = {'clevr_distance':'clevr_closest_object_distance', 'clevr_count':'clevr_count_all', 'smallnorb_elevation':'smallnorb_label_elevation','smallnorb_azimuth':'smallnorb_label_azimuth',
               'retinopathy':'diabetic_retinopathy', 'kitti_distance':'kitti_closest_vehicle_distance', 'flowers102':'flowers', 'dspr_y_position': 'dsprites_label_y_position', 'dspr_x_position':'dsprites_label_x_position', 'dspr_orientation':'dsprites_label_orientation'}
    if benchmark_name in mapping:
        download_name = mapping[benchmark_name]
    

    benchmark = HuggingFaceDataset(
        transform=None, dataset_url=f"haideraltahan/wds_{download_name}"
    )

    # Get class names from the benchmark dataset
    class_names = benchmark.classes
    templates = benchmark.templates

    if small_set:
        # Allowed values for small sets
        candidates = [32]
        num_classes = [c for c in candidates if c <= len(class_names)]
        if len(num_classes) == 0:
            num_classes.append(len(class_names))
    else:
        num_classes = [2**i for i in range(1, 40) if 2**i <= len(class_names)]
        # num_classes = [len(class_names)]
        if not num_classes or num_classes[-1] < len(class_names):
            num_classes.append(len(class_names))

    benchmark = partial(
        HuggingFaceDataset,
        dataset_url=f"haideraltahan/wds_{download_name}",
        max_num_samples=5000
    )
    bench_names = [f"{benchmark_name}_{num}" for num in num_classes]

    for num in num_classes:
        print(f"Adding benchmark: {benchmark_name}_{num} with {num} classes")
        evaluator.add_benchmark(
            benchmark_name=f"{benchmark_name}_{num}",
            benchmark=benchmark,
            handlers={
                "multi_choice_classification": partial(
                    MultiChoiceClassificationBenchmarkHandler,
                    class_names=class_names,
                    num_classes=num,
                ),
                "zeroshot_classification":partial(
                    ZeroShotBenchmarkHandler,
                    class_names=class_names,
                    templates=templates,
                    num_classes=num,
                ), 
            },
            meta_data={
                "benchmark_type": "object recognition",
            },
        )

    evaluator.update_benchmark_list(bench_names)

    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_classification", 'zeroshot_classification'],
    )


if __name__ == "__main__":
    fire.Fire(main)
