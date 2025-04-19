sbatch --array=0-19 $1 text_classification
sbatch --array=0-19 $1 clip_judge_classification
sbatch --array=0-19 $1 llm_judge_classification
sbatch --array=0-19 $1 clip_judge_relation