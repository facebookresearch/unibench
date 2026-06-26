#!/bin/bash
# run.sh – submit evaluation jobs, skipping any already completed.
#
# Classification: one SLURM array job per model; array tasks 0-44 = benchmark IDs.
#                 Done marker: <OUT_DIR>/<model>/<bench>_*.f  (any class-count suffix)
# Relation:       one regular job per model covering all 8 relation benchmarks.
#                 Done marker: <OUT_DIR>/<model>/<bench>.f for each benchmark
# VQA:            one regular job per model.
#                 Done marker: <OUT_DIR>/<model>/<bench>.f for each benchmark

NUM_MODELS=33   # last index (0-based)

RUN_DATE=$(date +%Y%m%d_%H%M%S)
LOG_DIR=./scripts_log/${RUN_DATE}
OUT_DIR=$(realpath ./script_outputs/)
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
LOG_FILE=${LOG_DIR}/slurm_%A_%a.out

echo "Logging to:        ${LOG_FILE}"
echo "Output directory:  ${OUT_DIR}"

cd /storage/home/hcoda1/6/haltahan6/scratch/unibench/eval_scripts/haider

# Load lists from eval.py once
readarray -t MODELS < <(python3 -c "from eval import MODELS; print('\n'.join(MODELS))")
readarray -t CLASS_BENCHMARKS < <(python3 -c "from eval import CLASSIFICATION_BENCHMARKS; print('\n'.join(CLASSIFICATION_BENCHMARKS))")
readarray -t REL_BENCHMARKS < <(python3 -c "from eval import RELATION_BENCHMARKS; print('\n'.join(RELATION_BENCHMARKS))")
readarray -t VQA_BENCH < <(python3 -c "from eval import VQA_BENCHMARKS; print('\n'.join(VQA_BENCHMARKS))")

# Returns 0 (true) if all given benchmarks have a .f file for the model
all_done() {
    local model_name=$1; shift
    for bench in "$@"; do
        [[ ! -f "${OUT_DIR}/${model_name}/${bench}.f" ]] && return 1
    done
    return 0
}

# Compact a newline-separated list of ints into a SLURM range string (e.g. "0-3,7,10-12")
compact_range() {
    python3 -c "
import sys
ids = sorted(int(x) for x in sys.stdin.read().split())
parts, i = [], 0
while i < len(ids):
    j = i
    while j + 1 < len(ids) and ids[j+1] == ids[j] + 1:
        j += 1
    parts.append(str(ids[i]) if i == j else f'{ids[i]}-{ids[j]}')
    i = j + 1
print(','.join(parts))
"
}

echo "Submitting classification jobs for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    model_name="${MODELS[$model_idx]}"

    todo_ids=()
    for bench_id in $(seq 0 44); do
        bench_name="${CLASS_BENCHMARKS[$bench_id]}"
        # Match any class-count suffix (e.g. caltech101_10.f, caltech101_4.f)
        if ! compgen -G "${OUT_DIR}/${model_name}/${bench_name}_"*.f > /dev/null 2>&1; then
            todo_ids+=($bench_id)
        fi
    done

    if [[ ${#todo_ids[@]} -eq 0 ]]; then
        echo "  [skip] ${model_name}: all classification benchmarks done"
        continue
    fi

    array_spec=$(printf '%s\n' "${todo_ids[@]}" | compact_range)
    echo "  [submit] ${model_name}: ${#todo_ids[@]}/45 benchmarks (array: ${array_spec})"
    sbatch --array="${array_spec}" --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_name}" "classification"
done

echo "Submitting relation jobs for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    model_name="${MODELS[$model_idx]}"
    if all_done "$model_name" "${REL_BENCHMARKS[@]}"; then
        echo "  [skip] ${model_name}: all relation benchmarks done"
        continue
    fi
    echo "  [submit] ${model_name}: relation"
    sbatch --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_name}" "relation"
done

echo "Submitting VQA jobs (openapps, mmmu_pro) for model indices 0-${NUM_MODELS} ..."
for model_idx in $(seq 0 ${NUM_MODELS}); do
    model_name="${MODELS[$model_idx]}"
    if all_done "$model_name" "${VQA_BENCH[@]}"; then
        echo "  [skip] ${model_name}: all VQA benchmarks done"
        continue
    fi
    echo "  [submit] ${model_name}: vqa"
    sbatch --output="${LOG_FILE}" \
        evaluation.sh "${OUT_DIR}" "${model_name}" "vqa"
done

echo "Done submitting."
