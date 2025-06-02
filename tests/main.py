from unibench import Evaluator
import fire


def main(num_workers=64, idx=0):
    evaluator = Evaluator(
        download_aggregate_precomputed=False,
        models='vllm',
        model_id=idx,
        num_workers=num_workers,
    )
    evaluator.evaluate(
        batch_per_gpu=4,
        tasks=["text_classification"],
    )


if __name__ == "__main__":
    fire.Fire(main)