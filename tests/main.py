from unibench import Evaluator
import fire


def main():
    evaluator = Evaluator(
        download_aggregate_precomputed=False,
        models=["llava_1_5_7b"],
    )
    evaluator.evaluate(
        batch_per_gpu=4,
        tasks=["text_classification"],
    )


if __name__ == "__main__":
    fire.Fire(main)