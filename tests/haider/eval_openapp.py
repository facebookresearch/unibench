from functools import partial

import fire

from unibench import Evaluator
from unibench.common_utils.constants import OUTPUT_DIR


SIGLIP2_MODELS = [
    # "siglip2_so400_14_378",
    "siglip2_so400_14",
    # "siglip2_so400_16_512",
    # "siglip2_so400_16_384",
    # "siglip2_so400_16_256",
    # "siglip2_vitL16_512",
    # "siglip2_vitL16_384",
]

THEMES = ["default", "dark_theme", "german", "challenging_font", "long_descriptions"]


def main(output_dir=OUTPUT_DIR, num_workers=8, idx=1, model_name=None):

    if model_name is not None:
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models=[model_name],
            num_workers=num_workers,
            benchmarks=[
                'openapps'
            ],
            output_dir=output_dir,
        )
    elif idx == 'all':
        evaluator = Evaluator(
            download_aggregate_precomputed=False,
            models="vllm",
            num_workers=num_workers,
            benchmarks=[
                'openapps'
            ],
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
                 'paligemma_3b_mix_448',
                 'gemma4_e2b',
                 'gemma4_e4b',
                 'gemma4_26b_a4b',
                 'gemma4_31b',
                 ],
            benchmarks=["openapps"],
            model_id=idx,
            num_workers=num_workers,
            output_dir=output_dir,
        )

    evaluator.evaluate(
        batch_per_gpu=8,
        tasks=["vqa_multiple_choice", "multi_choice_vqa"],
    )


if __name__ == "__main__":
    fire.Fire(main)
