from unibench import Evaluator
import fire

models = [
    "paligemma_3b_224",
    "paligemma_3b_mix_224",
    "paligemma_3b_448",
    "paligemma_3b_mix_448",
    "paligemma2_3b_mix_224",
    "paligemma2_3b_mix_448",
    "llava_1_5_7b",
    "bakllava_1_7b",
    "chameleon_7b",
    "llava_next_vicuna_7b",
    "paligemma2_10b_mix_224",
    "paligemma2_10b_mix_448",
    "llama_3_2_11b",
    "llava_1_5_13b",
    "llava_next_vicuna_13b",
    "llava_next_llama_8b",
    "llava_next_mistral_7b",
    "paligemma2_28b_mix_224",
    "paligemma2_28b_mix_448",
    "llava_next_34b",
    "gemma3_27b",
    "gemma3_12b",
    "gemma3_4b",
    "aya_8b",
    "phi_4",
    "llama_4_maverick",
    "llama_4_scout",
    "llama_3_2_11b_cot",
    # "cambrian_8b",
    # "chameleon_30b",
    # "llava_1_6_34b",
    # "llama_4_scout",
    # "llava_next_110b", not enough memory
]


def main(task_name="text_classification", idx=2):
    evaluator = Evaluator(
        download_aggregate_precomputed=False, model_id=idx, models=models
    )

    print("-" * 20)
    print(f"Evaluating {models[idx]} on {task_name}")
    print("-" * 20)

    evaluator.evaluate(
        batch_per_gpu=2,
        tasks=[task_name],
    )


if __name__ == "__main__":
    fire.Fire(main)
