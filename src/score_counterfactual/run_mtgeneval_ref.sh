#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/nonambiguous-counterfactual/mtgeneval/references"


# Array of languages
LANGS=("it" "es" "de" "pt" "ar" "fr" "hi" "ru")

# Array of models with their IDs
declare -A MODELS=(
    ["Kiwi 22"]="Unbabel/wmt22-cometkiwi-da"
    # ["Kiwi 23 XL"]="Unbabel/wmt23-cometkiwi-da-xl"
    # ["Kiwi 23 XXL"]="Unbabel/wmt23-cometkiwi-da-xxl"
    # ["xCOMET XL"]="Unbabel/XCOMET-XL"
    # ["xCOMET XXL"]="Unbabel/XCOMET-XXL"
    # ["MetricX23 LARGE"]="google/metricx-23-qe-large-v2p0"
    # ["MetricX23 XL"]="google/metricx-23-qe-xl-v2p0"
    # ["Llama 3.1 70B"]="meta-llama/Llama-3.1-70B-Instruct"
    # ["Mistral 7B"]="mistralai/Mistral-7B-Instruct-v0.2"
    ["Gemma 2 9B"]="google/gemma-2-9b-it"
)

# Loop through each language
for lang in "${LANGS[@]}"; do
    # Loop through each model
    for model_name in "${!MODELS[@]}"; do
        model_id="${MODELS[$model_name]}"
        model_name=$(echo "$model_id" | sed 's/\//--/g')
        # Check if scores already exist
        scores_path="$BASE_OUTPUT_DIR/$lang/$model_name/scores.csv"
        if [ -f "$scores_path" ]; then
            echo "Skipping $lang with $model_name - scores already exist at $scores_path"
            continue
        fi

        echo "About to run the scoring script for: $lang with $model_id"
        
        # Run the scoring script
        python src/score_counterfactual/score_mtgeneval.py \
            --qe_metric_name_or_path "$model_id" \
            --tgt_lang "$lang" \
            --output_dir "$BASE_OUTPUT_DIR"
    done
done

echo "All scoring tasks completed!"