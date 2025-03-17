from functools import partial
from unibench import Evaluator
from unibench.benchmarks_zoo import TextClassificationBenchmarkHandler

from unibench.benchmarks_zoo.wrappers.huggingface import HuggingFaceDataset

benchmark = partial(HuggingFaceDataset, dataset_url="haideraltahan/wds_imagenet1k")

handler = partial(
    TextClassificationBenchmarkHandler,
    benchmark_name="imagenet1k_text_5",
    class_names=HuggingFaceDataset(dataset_url="haideraltahan/wds_imagenet1k").classes,
    num_classes=5,
)

eval = Evaluator()

eval.add_benchmark(
    benchmark,
    handler,
    meta_data={
        "benchmark_type": "object recognition",
    },
)
eval.update_benchmark_list(["imagenet1k_text_5"])
eval.update_model_list(["llava_1_5_7b"])
eval.evaluate()