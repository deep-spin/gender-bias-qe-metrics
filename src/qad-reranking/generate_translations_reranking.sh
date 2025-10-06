#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

BASE_OUTPUT_DIR="$ROOT_DIR/results/translations-qad-ambiguous/"


# Array of languages
LANGS=("it" "es" "de" "pt"  "fr"  "ru")

model_name_path="Unbabel/TowerInstruct-7B-v0.2"

declare -A METRICS=(
    ["Kiwi 22"]="Unbabel/wmt22-cometkiwi-da"
    ["Kiwi 23 XXL"]="Unbabel/wmt23-cometkiwi-da-xxl"
)

# Loop through each language
for lang in "${LANGS[@]}"; do
    for metric in "${METRICS[@]}"; do
        model_name=$(echo "$model_name_path" | sed 's/\//--/g')
        # Check if scores already exist
        generations_path="$BASE_OUTPUT_DIR/$lang/$model_name/reranking/hypothesis.json"
        if [ -f "$generations_path" ]; then
            echo "Skipping $lang with $model_name - generations already exist at $generations_path"
        else
            echo "Running the generations for: $lang with $model_name_path"
            
            # Run the scoring script
            CUDA_VISIBLE_DEVICES=0 \
            python src/qad-reranking/translate.py \
                --model_name_or_path "$model_name_path" \
                --tgt_lang "$lang" \
                --output_path "$generations_path" \
                --max_new_tokens 1024 \
                --temperature 0.7 \
                --num_return_sequences 50 \
                --epsilon_cutoff 0.2 \
                --dry_run
        fi
        # Reranking
        metric_name=$(echo "$metric" | sed 's/\//--/g')
        out_path="$BASE_OUTPUT_DIR/$lang/$model_name/reranking/$metric_name/generations.json"
        if [ -f "$out_path" ]; then
            echo "Skipping $lang with $metric_name - reranking already exist at $out_path"
        else
            echo "Running the reranking for: $lang with $metric_name"
            # Run the reranking script
            CUDA_VISIBLE_DEVICES=0 \
                python src/qad-reranking/score_rerank.py \
                --model_name_or_path "$metric" \
                --hypothesis_path "$generations_path" \
                --output_path "$out_path" \
                --batch_size 32
        fi
    done
done

echo "All scoring tasks completed!"