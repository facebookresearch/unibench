from unibench import Evaluator
import fire


def main():
    evaluator = Evaluator(
        benchmarks=["sugarcrepe"],
        download_aggregate_precomputed=False,
        models=["blip2_7b"],
    )
    evaluator.evaluate(
        batch_per_gpu=4,
        tasks=["clip_judge_relation"],
    )


if __name__ == "__main__":
    fire.Fire(main)
