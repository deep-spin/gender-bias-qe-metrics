#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

BASE_OUTPUT_DIR="$ROOT_DIR/results/translations-qad-ambiguous/"



# Array of languages
LANGS=("it" "es" "de" "pt"  "fr"  "ru")

model_name_path="Unbabel/TowerInstruct-7B-v0.2"


# Loop through each language
for lang in "${LANGS[@]}"; do
    model_name=$(echo "$model_name_path" | sed 's/\//--/g')
    # Check if scores already exist
    generations_path="$BASE_OUTPUT_DIR/$lang/$model_name/greedy/generations.json"
    if [ -f "$generations_path" ]; then
        echo "Skipping $lang with $model_name - generations already exist at $generations_path"
        continue
    fi

        echo "About to run the generations for: $lang with $model_name_path"
        
        # Run the scoring script
        CUDA_VISIBLE_DEVICES=0 \
        python src/qad-reranking/translate.py \
            --model_name_or_path "$model_name_path" \
            --tgt_lang "$lang" \
            --output_path "$generations_path" \
            --max_new_tokens 1024 \
            --temperature 1.0 \
            --topk 1
done

echo "All scoring tasks completed!"