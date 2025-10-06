#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/nonambiguous-contextual/mtgeneval"


# Array of languages
LANGS=("it" "es" "de" "pt" "ar" "fr" "hi" "ru")


# if TRANSLATE_CONTEXT is True, then the contexts will be translated using the translation_model_name
# and the context_translations_dir will be used to load the translated contexts
TRANSLATE_CONTEXT=True
context_translations_dir="$ROOT_DIR/data/mtgeneval/translated-contexts-paper/google-translate"
translation_model_name="google-translate"


# if TRANSLATE_CONTEXT is False, then the original contexts will be loaded.
# so you dont need to provide the context_translations_dir and translation_model_name

# TRANSLATE_CONTEXT=False

# Array of models with their IDs
declare -A MODELS=(
    ["Kiwi 22"]="Unbabel/wmt22-cometkiwi-da"
    # ["Kiwi 23 XL"]="Unbabel/wmt23-cometkiwi-da-xl"
    # ["Kiwi 23 XXL"]="Unbabel/wmt23-cometkiwi-da-xxl"
    # ["xCOMET XL"]="Unbabel/XCOMET-XL"
    # ["xCOMET XXL"]="Unbabel/XCOMET-XXL"
    # ["MetricX23 LARGE"]="google/metricx-23-qe-large-v2p0"
    # ["MetricX23 XL"]="google/metricx-23-qe-xl-v2p0"
)


# Loop through each language
for lang in "${LANGS[@]}"; do
    # Loop through each model
    for model_name in "${!MODELS[@]}"; do
        model_id="${MODELS[$model_name]}"
        model_name=$(echo "$model_id" | sed 's/\//--/g')

        echo "About to run the scoring script for: $lang with $model_id"

        if [ "$TRANSLATE_CONTEXT" = True ]; then
            # Check if scores already exist
            scores_path="$BASE_OUTPUT_DIR/$lang/$model_name/with-translated-contexts/$translation_model_name/scores.csv"
            if [ -f "$scores_path" ]; then
                echo "Skipping $lang with $model_name - scores already exist at $scores_path"
                continue
            fi
            context_translations_path="$context_translations_dir/translated-context_$lang.txt"
            python src/score_contextual/score_mtgeneval.py \
                --qe_metric_name_or_path "$model_id" \
                --tgt_lang "$lang" \
                --output_dir "$BASE_OUTPUT_DIR" \
                --context_translations_path "$context_translations_path" \
                --translation_model_name "$translation_model_name" \
                --metric_type "neural"
        else
            # Check if scores already exist
            scores_path="$BASE_OUTPUT_DIR/$lang/$model_name/with-original-contexts/scores.csv"
            if [ -f "$scores_path" ]; then
                echo "Skipping $lang with $model_name - scores already exist at $scores_path"
                continue
            fi
            python src/score_contextual/score_mtgeneval.py \
                --qe_metric_name_or_path "$model_id" \
                --tgt_lang "$lang" \
                --output_dir "$BASE_OUTPUT_DIR" \
                --metric_type "neural"
        fi
    done
done

echo "All scoring tasks completed!"