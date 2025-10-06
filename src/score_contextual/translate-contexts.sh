#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

BASE_OUTPUT_DIR="$ROOT_DIR/data/mtgeneval/translated-contexts-paper"
BATCH_SIZE=32

# Array of languages
LANGS=("it" "es" "de" "pt" "ar" "fr" "hi" "ru")

# Array of models with their IDs
declare -A MODELS=(
    ["nllb-200-3.3B"]="facebook/nllb-200-3.3B"
)

# Loop through each language
for lang in "${LANGS[@]}"; do
    # Loop through each model
    for model_name in "${!MODELS[@]}"; do
        model_id="${MODELS[$model_name]}"
        model_name=$(echo "$model_id" | sed 's/\//--/g')
        

        echo "About to run the translation script for: $lang with $model_id"
        
        # Check if model is nllb and set vllm_usage flag accordingly
        if [[ "$model_id" == *"nllb"* ]]; then
            # echo "Using HuggingFace pipeline for NLLB model"
            python src/score_contextual/translate_contexts.py \
                --model_name_or_path "$model_id" \
                --tgt_lang "$lang" \
                --output_dir "$BASE_OUTPUT_DIR" \
                --hf_usage \
                --batch_size "$BATCH_SIZE"
        else
            # echo "Using VLLM for non-NLLB model" 
            python src/score_contextual/translate_contexts.py \
                --model_name_or_path "$model_id" \
                --tgt_lang "$lang" \
                --output_dir "$BASE_OUTPUT_DIR" \
                --batch_size "$BATCH_SIZE"
        fi
    done
done

echo "All translation tasks completed!"