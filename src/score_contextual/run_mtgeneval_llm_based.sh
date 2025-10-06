#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/nonambiguous-contextual/mtgeneval"


# Array of languages
LANGS=("it" "es" "de" "pt" "ar" "fr" "hi" "ru")

GEMBA_SRC_CTX_TYPE="standard" # ["standard", "paraphrased"] choose between these two options

# Array of models with their IDs
declare -A MODELS=(
    # ["Llama 3.1 70B"]="meta-llama/Llama-3.1-70B-Instruct"
    ["Mistral 7B"]="mistralai/Mistral-7B-Instruct-v0.2"
    # ["Gemma 2 9B"]="google/gemma-2-9b-it"
)


# Loop through each language
for lang in "${LANGS[@]}"; do
    # Loop through each model
    for model_name in "${!MODELS[@]}"; do
        model_id="${MODELS[$model_name]}"
        model_name=$(echo "$model_id" | sed 's/\//--/g')

        # Check if scores already exist
        scores_path="$BASE_OUTPUT_DIR/$lang/$model_name/$GEMBA_SRC_CTX_TYPE/scores.csv"
        if [ -f "$scores_path" ]; then
            echo "Skipping $lang with $model_name - scores already exist at $scores_path"
            continue
        fi

        echo "About to run the scoring script for: $lang with $model_id"

        python src/score_contextual/score_mtgeneval.py \
            --qe_metric_name_or_path "$model_id" \
            --tgt_lang "$lang" \
            --output_dir "$BASE_OUTPUT_DIR" \
            --metric_type "llm" \
            --gemba_src_ctx_type "$GEMBA_SRC_CTX_TYPE"
    done
done

echo "All scoring tasks completed!"