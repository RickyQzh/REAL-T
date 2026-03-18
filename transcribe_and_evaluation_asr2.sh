#!/bin/bash

set -eu

# ==== FireRedASR2-AED: local Python inference for all datasets ====
# Independent from transcribe_and_evaluation.sh: that script uses FireRedASR-AED-L / whisper;
# this one uses FireRedASR2-AED for all datasets.
# Outputs: {BASE_DIR}/{dataset}/FireRedASR2-AED/predicted.csv, and
#          {BASE_DIR}/{BASE_NAME}_TER_ASR2_AED.txt / .csv

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

# Python scripts
ASR2_SCRIPT="./asr/asr_inference_fireredasr2.py"
EVAL_SCRIPT="./utils/asr_evaluation.py"

CHINESE_ASR2_MODEL="FireRedASR2-AED"
ENGLISH_ASR_MODEL="whisper-large-v2"

# Test set (PRIMARY or BASE)
TEST_SET_DIR="./datasets/REAL-T/PRIMARY"

# Local FireRedASR2S settings
FIREREDASR2S_ROOT="./FireRedASR2S"
FIREREDASR2_MODEL_DIR="${FIREREDASR2S_ROOT}/pretrained_models/FireRedASR2-AED"
USE_GPU=1
USE_HALF=0
ASR_BATCH_SIZE=1

# Base directories to process (same as main script; e.g. ./output/PRIMARY/BSRNN)
BASE_DIRS=(
    "./output/PRIMARY/BSRNN"
)

# Usage: bash -i ./transcribe_and_evaluation_asr2.sh 1       (ASR2 for all datasets)
#        bash -i ./transcribe_and_evaluation_asr2.sh 2       (only Evaluation -> TER_ASR2_AED)
#        bash -i ./transcribe_and_evaluation_asr2.sh 1 2     (both)

MAPPING_CSV_NAME="tse_audio_mapping.csv"
MODES=("$@")

if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (ASR2 for all datasets), 2 (Evaluation -> TER_ASR2_AED), or both."
    exit 1
fi

# ==== Functions ====

run_asr2_all_datasets() {
    if [ ! -d "$FIREREDASR2S_ROOT" ]; then
        echo "FireRedASR2S root not found: $FIREREDASR2S_ROOT"
        exit 1
    fi
    if [ ! -d "$FIREREDASR2_MODEL_DIR" ]; then
        echo "FireRedASR2-AED model directory not found: $FIREREDASR2_MODEL_DIR"
        exit 1
    fi

    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Processing base directory (FireRedASR2-AED, all datasets): $BASE_DIR"
        mapfile -t dataset_dirs < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
        if [ ${#dataset_dirs[@]} -eq 0 ]; then
            echo "No datasets found under $BASE_DIR, skipping."
            continue
        fi
        for dataset_path in "${dataset_dirs[@]}"; do
            [ -d "$dataset_path" ] || continue
            dataset=$(basename "$dataset_path")
            TSE_MAPPING_CSV="${dataset_path}/${MAPPING_CSV_NAME}"
            PREDICTED_DIR="${dataset_path}/${CHINESE_ASR2_MODEL}"
            if [ ! -f "$TSE_MAPPING_CSV" ]; then
                echo "Mapping not found: $TSE_MAPPING_CSV, skipping dataset $dataset."
                continue
            fi
            echo "Running FireRedASR2-AED for dataset: $dataset"
            echo "  Output directory: $PREDICTED_DIR"
            mkdir -p "$PREDICTED_DIR"
            python3 "$ASR2_SCRIPT" \
                --audio_mapping "$TSE_MAPPING_CSV" \
                --output_dir "$PREDICTED_DIR" \
                --dataset_name "$dataset" \
                --fireredasr2s_root "$FIREREDASR2S_ROOT" \
                --model_dir "$FIREREDASR2_MODEL_DIR" \
                --use_gpu "$USE_GPU" \
                --use_half "$USE_HALF" \
                --batch_size "$ASR_BATCH_SIZE"
        done
    done
    echo "FireRedASR2-AED (all datasets) ASR completed!"
}

run_evaluation_ter_asr2() {
    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Running TER Evaluation (FireRedASR2-AED -> TER_ASR2_AED)..."
        BASE_NAME=$(basename "$BASE_DIR")
        RESULT_TXT="${BASE_DIR}/${BASE_NAME}_TER_ASR2_AED.txt"
        RESULT_CSV="${BASE_DIR}/${BASE_NAME}_TER_ASR2_AED.csv"
        (
            python3 "$EVAL_SCRIPT" \
                --ground_truth_dir "$TEST_SET_DIR" \
                --save_path "$RESULT_CSV" \
                --predicted_dir "$BASE_DIR" \
                --chinese_asr_model "$CHINESE_ASR2_MODEL" \
                --english_asr_model "$CHINESE_ASR2_MODEL"
        ) > "$RESULT_TXT"
        echo "Evaluation (TER_ASR2_AED) completed: $RESULT_TXT, $RESULT_CSV"
    done
}

# ==== Execution ====

for mode in "${MODES[@]}"; do
    if [ "$mode" == "1" ]; then
        run_asr2_all_datasets
    elif [ "$mode" == "2" ]; then
        run_evaluation_ter_asr2
    else
        echo "Invalid mode: $mode. Please use 1 (ASR2 for all datasets), 2 (Evaluation -> TER_ASR2_AED), or both."
        exit 1
    fi
done

echo "All tasks finished successfully!"
