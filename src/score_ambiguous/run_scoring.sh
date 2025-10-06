#!/bin/bash

ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv

source $Path_to_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATASET=mtgeneval

# Set languages and base output directory based on dataset
case $DATASET in
    "gate")
        LANGUAGES=("it")
        BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/ambiguous/gate"
        SCRIPT="src/score_ambiguous/score_gate.py"
        ;;
    "mgente")
        LANGUAGES=("it" "es" "de")
        BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/ambiguous/mgente"
        SCRIPT="src/score_ambiguous/score_mgente.py"
        ;;
    "mtgeneval")
        LANGUAGES=("it" "es" "de" "pt" "ar" "fr" "hi" "ru")
        BASE_OUTPUT_DIR="$ROOT_DIR/results/scores/ambiguous/mtgeneval"
        SCRIPT="src/score_ambiguous/score_mtgeneval.py"
        ;;
    *)
        echo "Invalid dataset name. Please use one of: GATE, mGeNTE, MT-GenEval"
        exit 1
        ;;
esac

# Common parameters
NUM_GPUS=1
# BATCH_SIZE=4

# Array of models with their IDs
declare -A MODELS=(
    ["Kiwi 22"]="Unbabel/wmt22-cometkiwi-da"
    ["Kiwi 23 XL"]="Unbabel/wmt23-cometkiwi-da-xl"
    ["Kiwi 23 XXL"]="Unbabel/wmt23-cometkiwi-da-xxl"
    ["xCOMET XL"]="Unbabel/XCOMET-XL"
    # ["xCOMET XXL"]="Unbabel/XCOMET-XXL"
    ["MetricX23 LARGE"]="google/metricx-23-qe-large-v2p0"
    #["MetricX23 XL"]="google/metricx-23-qe-xl-v2p0"
    # ["Llama 3.1 70B"]="meta-llama/Llama-3.1-70B-Instruct"
    ["Mistral 7B"]="mistralai/Mistral-7B-Instruct-v0.2"
    ["Gemma 2 9B"]="google/gemma-2-9b-it"
)

# Process each language and model combination
for lang in "${LANGUAGES[@]}"; do
    for model_name in "${!MODELS[@]}"; do
        # Create output directory
        model_path="${MODELS[$model_name]}"
        model_path_replaced=$(echo "$model_path" | sed 's/\//--/g')

        # Check if scores already exist
        scores_path="$BASE_OUTPUT_DIR/$lang/$model_path_replaced/scores.csv"
        if [ -f "$scores_path" ]; then
            echo "Skipping $lang with $model_name - scores already exist at $scores_path"
            continue
        fi

        echo "About to run the scoring script for: $lang with  $model_name"
        
        # Run the scoring script
        python3 $SCRIPT \
            --qe_metric_name_or_path "$model_path" \
            --tgt_lang "$lang" \
            --output_dir "$BASE_OUTPUT_DIR"
       
        # Add a small delay between runs to prevent overwhelming the system
        sleep 10
    done
done