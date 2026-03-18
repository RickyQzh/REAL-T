#!/bin/bash

set -eu

# ==== Configurations ====

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

# Python scripts
ASR_SCRIPT="./asr/asr_inference.py"
EVAL_SCRIPT="./utils/asr_evaluation.py"

# ASR model names
CHINESE_ASR_MODEL="FireRedASR-AED-L"
ENGLISH_ASR_MODEL="whisper-large-v2"

# Test Which PRIMARY test set or BASE test set or other
TEST_SET_DIR="./datasets/REAL-T/PRIMARY"
# TEST_SET_DIR="./datasets/REAL-T/BASE"

INCLUDING_FISHER="False"

# Datasets that use Chinese ASR vs English ASR (space-separated)
CHINESE_DATASETS="${CHINESE_DATASETS:-AliMeeting AISHELL-4}"
ENGLISH_DATASETS="${ENGLISH_DATASETS:-AMI DipCo CHiME6 Fisher}"

# Base directories to process
BASE_DIRS=(
    "./output/PRIMARY/bsrnn_vox1"
)

# Get operation mode from arguments
# Usage: bash -i ./transcribe_and_evaluation.sh 1       (only ASR)
#        bash -i ./transcribe_and_evaluation.sh 2       (only Evaluation)
#        bash -i ./transcribe_and_evaluation.sh 1 2     (both ASR and Evaluation)

# Mapping csv filename
MAPPING_CSV_NAME="tse_audio_mapping.csv"

MODES=("$@")

if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (ASR), 2 (Evaluation), or both."
    exit 1
fi

# ==== Functions ====

run_asr() {
    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Processing base directory: $BASE_DIR"

        for dataset_path in "$BASE_DIR"/*; do
            mapfile -t dataset_dirs < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
            if [ ${#dataset_dirs[@]} -eq 0 ]; then
                echo "No datasets found under $BASE_DIR, skipping."
                continue
            fi
            echo "Found ${#dataset_dirs[@]} datasets under $BASE_DIR:"
            for dir in "${dataset_dirs[@]}"; do
                echo "  - $(basename "$dir")"
            done

            if [ -d "$dataset_path" ]; then
                dataset=$(basename "$dataset_path")
                # Set ASR model according to dataset (CHINESE_DATASETS / ENGLISH_DATASETS)
                ASR_MODEL_NAME=""
                if [[ " ${CHINESE_DATASETS} " =~ " ${dataset} " ]]; then
                    ASR_MODEL_NAME="$CHINESE_ASR_MODEL"
                elif [[ " ${ENGLISH_DATASETS} " =~ " ${dataset} " ]]; then
                    ASR_MODEL_NAME="$ENGLISH_ASR_MODEL"
                fi
                if [[ -z "$ASR_MODEL_NAME" ]]; then
                    echo "Dataset $dataset is not in CHINESE_DATASETS or ENGLISH_DATASETS. Skipping..."
                    continue
                fi

                TSE_MAPPING_CSV="${dataset_path}/${MAPPING_CSV_NAME}"
                PREDICTED_DIR="${dataset_path}/${ASR_MODEL_NAME}"

                echo "Running ASR for dataset: $dataset"
                echo "Audio mapping file: $TSE_MAPPING_CSV"
                echo "Output directory: $PREDICTED_DIR"

                mkdir -p "$PREDICTED_DIR"

                python3 "$ASR_SCRIPT" \
                    --audio_mapping "$TSE_MAPPING_CSV" \
                    --model_name "$ASR_MODEL_NAME" \
                    --dataset_name "$dataset" \
                    --output_dir "$PREDICTED_DIR"
            fi
        done
    done
    echo "ASR processing completed!"
}

run_asr_evaluation() {
    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Running TER Evaluation..."
        BASE_NAME=$(basename "$BASE_DIR")
        # echo "$BASE_NAME"
        mapfile -t dataset_dirs < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
        if [ ${#dataset_dirs[@]} -eq 0 ]; then
            echo "No datasets found under $BASE_DIR, skipping."
            continue
        fi
        echo "Found ${#dataset_dirs[@]} datasets under $BASE_DIR:"
        for dir in "${dataset_dirs[@]}"; do
            echo "  - $(basename "$dir")"
        done

        if [ "$INCLUDING_FISHER" == "True" ]; then
            SUFFIX="_including_fisher"
        else
            SUFFIX=""
        fi

        RESULT_TXT="${BASE_DIR}/${BASE_NAME}_TER${SUFFIX}.txt"
        RESULT_CSV="${BASE_DIR}/${BASE_NAME}_TER${SUFFIX}.csv"

        (
            python3 "$EVAL_SCRIPT" \
                --ground_truth_dir "$TEST_SET_DIR" \
                --save_path "$RESULT_CSV" \
                --predicted_dir "$BASE_DIR" \
                --chinese_asr_model "$CHINESE_ASR_MODEL" \
                --english_asr_model "$ENGLISH_ASR_MODEL" \
                $( [ "$INCLUDING_FISHER" == "True" ] && echo "--include_fisher" )
        ) > "$RESULT_TXT"

        echo "Evaluation completed!"
    done
}


# ==== Execution ====

for mode in "${MODES[@]}"; do
    if [ "$mode" == "1" ]; then
        run_asr
    elif [ "$mode" == "2" ]; then
        run_asr_evaluation
    else
        echo "Invalid mode: $mode. Please use 1 (ASR), 2 (Evaluation), or both."
        exit 1
    fi
done

echo "All tasks finished successfully!"
