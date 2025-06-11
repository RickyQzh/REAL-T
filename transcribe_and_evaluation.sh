#!/bin/bash

set -eu

# ==== Configurations ====

# Activate conda environment
conda activate REAL-T

# Python scripts
ASR_SCRIPT="./asr/asr_inference.py"
EVAL_SCRIPT="./utils/asr_evaluation.py"

# ASR model names
CHINESE_ASR_MODEL="FireRedASR-AED-L"
ENGLISH_ASR_MODEL="whisper-large-v2"

# Test Which PRIMARY test set or BASE test set or other
TEST_SET_DIR="./datasets/REAL-T/PRIMARY"

# Mapping csv filename
MAPPING_CSV_NAME="tse_audio_mapping.csv"

INCLUDING_FISHER="False"

# Base directories to process
BASE_DIRS=(
    # "/root/shared-nvme/open-source/REAL-T/output/PRIMARY/bsrnn_hr_vox1"
    "/root/shared-nvme/open-source/debug/result_PRIMARY/tselm"
)

# Get operation mode from arguments
# Usage: ./this_script.sh 1       (only ASR)
#        ./this_script.sh 2       (only Evaluation)
#        ./this_script.sh 1 2     (both ASR and Evaluation)

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
            if [ -d "$dataset_path" ]; then
                dataset=$(basename "$dataset_path")

                # Set ASR model according to dataset
                if [[ "$dataset" == "AliMeeting" || "$dataset" == "AISHELL-4" ]]; then
                    ASR_MODEL_NAME="$CHINESE_ASR_MODEL"
                elif [[ "$dataset" == "AMI" || "$dataset" == "DipCo" || "$dataset" == "CHiME6" || "$dataset" == "Fisher" ]]; then
                    ASR_MODEL_NAME="$ENGLISH_ASR_MODEL"
                else
                    echo "Dataset $dataset is not supported. Skipping..."
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
        echo "$BASE_NAME"

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
