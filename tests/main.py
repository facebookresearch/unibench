from unibench import Evaluator
import fire


def main(num_workers=8, idx=1, model_name=None):

    if model_name is not None:
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models=[model_name],
            num_workers=num_workers,
        )
    elif idx == 'all':
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models="vllm",
            num_workers=num_workers,
        )
    else:
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models="vllm",
            model_id=idx,
            num_workers=num_workers,
        )
    evaluator.evaluate(
        batch_per_gpu=4,
        tasks=["text_classification"],
    )


if __name__ == "__main__":
    fire.Fire(main)
