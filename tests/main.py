from unibench import Evaluator
import fire

from unibench.common_utils.constants import OUTPUT_DIR


def main(output_dir=OUTPUT_DIR, num_workers=8, idx=1, model_name=None):

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
            models="vllm",
            model_id=idx,
            num_workers=num_workers,
            output_dir=output_dir,
        )
    evaluator.evaluate(
        batch_per_gpu=2,
        tasks=["multi_choice_classification", "multi_choice_relation"],
    )


if __name__ == "__main__":
    fire.Fire(main)
