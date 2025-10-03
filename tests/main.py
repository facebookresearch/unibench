from unibench import Evaluator
import fire

from unibench.benchmarks_zoo.wrappers.huggingface import HuggingFaceDataset
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
            models=['llama_3_2_11b_vision_instruct', 'llava_1_5_13b', 'llava_1_5_7b', 
                 'llava_1_6_mistral_7b',
                 'llava_1_6_vicuna_13b', 'llava_1_6_vicuna_7b', 'llava_next_llama_8b', 'aya_vision_32b', 'aya_vision_8b', 'gemma3_27b', 'gemma3_4b', 'gemma3_12b',
                 'paligemma2_10b_mix_224', 'paligemma2_10b_mix_448', 'paligemma2_28b_mix_224', 
                 'paligemma2_28b_mix_448', 'paligemma2_3b_mix_224', 'paligemma2_3b_mix_448', 
                'paligemma_3b_mix_224', 'paligemma_3b_mix_448'],
            model_id=idx,
            num_workers=num_workers,
            output_dir=output_dir,
        )
    
    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["multi_choice_classification", "multi_choice_relation"],
    )


if __name__ == "__main__":
    fire.Fire(main)
