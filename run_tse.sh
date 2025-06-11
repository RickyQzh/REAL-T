#!/bin/bash

conda activate REAL-T

# v1 is suitable for USEF-TFGridnet, bsrnn_vox1, bsrnn_hr_vox1, bsrnn_hr_100, bsrnn_hr_360, bsrnn_hr_SDR_360
# v2 is suitable for bsrnn_100,bsrnn_360, 

# Rename your model
# MODEL_NAME="bsrnn_hr_vox1"  # bsrnn_hr pretrained on VoxCeleb1
MODEL_NAME="bsrnn_vox1"     # bsrnn pretrained on VoxCeleb1

# Dataset names
DATASETS=("AliMeeting" "AMI" "CHiME6" "AISHELL-4" "DipCo")
# DATASETS=("AliMeeting" "AMI" "CHiME6" "AISHELL-4" "DipCo" "Fisher")

# Test subset
TEST_SET="PRIMARY"
# TEST_SET="BASE"

DEVICE="cuda"

# Base paths
BASE_META_PATH="./datasets/REAL-T"
BASE_OUTPUT_DIR="./output"


TSE_SCRIPT="./tse_baseline/tse.py"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TEST_SET}"
mkdir -p "$BASE_OUTPUT_DIR"
# Iterate over each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    META_CSV_PATH="${BASE_META_PATH}/${TEST_SET}/${DATASET}_meta.csv"
    UTTERANCE_MAP_CSV="${BASE_META_PATH}/mapping.csv"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}/"

    mkdir -p "$OUTPUT_DIR"

    echo "Starting TSE processing for $DATASET..."
    python3 "$TSE_SCRIPT" \
        --meta_csv_path "$META_CSV_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --utterance_map_csv "$UTTERANCE_MAP_CSV" \
        --device "$DEVICE" \
        --model_name "$MODEL_NAME"

    # Check if TSE processing succeeded
    if [ $? -ne 0 ]; then
        echo "TSE processing failed for $DATASET."
        exit 1
    fi
    echo "TSE processing completed for $DATASET."
done

echo "All datasets processed successfully!"