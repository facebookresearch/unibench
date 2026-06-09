import fire

from unibench import Evaluator
from unibench.common_utils.constants import OUTPUT_DIR


def main(model_name, benchmark_name, output_dir=OUTPUT_DIR, num_workers=0, batch_per_gpu=8):
    evaluator = Evaluator(
        download_aggregate_precomputed=False,
        models=[model_name],
        num_workers=num_workers,
        benchmarks=[benchmark_name],
        output_dir=output_dir,
    )

    evaluator.evaluate(
        batch_per_gpu=batch_per_gpu,
        tasks=["vqa_multiple_choice", "multi_choice_vqa"],
    )


if __name__ == "__main__":
    fire.Fire(main)
