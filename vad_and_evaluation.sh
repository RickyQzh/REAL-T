#!/bin/bash

set -eu

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_setup.sh"

# Python scripts
VAD_SCRIPT="./utils/vad_inference_firered.py"
PREPARE_LABEL_SCRIPT="./utils/prepare_label_segments.py"
EVAL_SCRIPT="./utils/tse_timing_evaluation.py"
VIS_SCRIPT="./utils/plot_tse_timeline.py"

# Test set (PRIMARY or BASE)
TEST_SET_DIR="./datasets/REAL-T/PRIMARY"

# GT overlap json: copy from REAL-T-Ext output to this path (see README)
GT_JSON_BASE_DIR="./datasets/REAL-T/json"

# Metadata dir for gender info
METADATA_DIR="./datasets/REAL-T/metadata"

# FireRedASR2S repo root (clone under REAL-T, contains fireredasr2s package)
FIREREDASR2S_ROOT="./FireRedASR2S"

# FireRedVAD model directory
FIRERED_VAD_MODEL_DIR="${FIREREDASR2S_ROOT}/pretrained_models/FireRedVAD/VAD"

# Base directories to process
BASE_DIRS=(
    "./output/PRIMARY/bsrnn_vox1"
)

# Only process these datasets (same as transcribe_and_evaluation.sh)
DATASETS=(AliMeeting AISHELL-4 AMI DipCo CHiME6 Fisher)

# Naming
MAPPING_CSV_NAME="tse_audio_mapping.csv"
VAD_DIR_NAME="FireRedVAD"
VAD_JSONL_NAME="vad_segments.jsonl"

# VAD / timing metric configs
USE_GPU=1
SPEECH_THRESHOLD=0.5
FRAME_SHIFT=0.01
COLLAR=0.05
MATCH_TOLERANCE=0.02

# Usage:
#   bash -i ./vad_and_evaluation.sh 1
#   bash -i ./vad_and_evaluation.sh 2
#   bash -i ./vad_and_evaluation.sh 3 (Visualization)
#   bash -i ./vad_and_evaluation.sh 1 2
MODES=("$@")

if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (VAD), 2 (Timing Evaluation), 3 (Visualization), or combination."
    exit 1
fi

run_vad() {
    if [ ! -d "$FIREREDASR2S_ROOT" ]; then
        echo "FireRedASR2S root not found: $FIREREDASR2S_ROOT"
        exit 1
    fi
    if [ ! -d "$FIRERED_VAD_MODEL_DIR" ]; then
        echo "FireRedVAD model directory not found: $FIRERED_VAD_MODEL_DIR"
        echo "Please set FIRERED_VAD_MODEL_DIR to your local FireRedVAD model path."
        exit 1
    fi

    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Running VAD for base directory: $BASE_DIR"
        mapfile -t dataset_dirs < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
        if [ ${#dataset_dirs[@]} -eq 0 ]; then
            echo "No datasets found under $BASE_DIR, skipping."
            continue
        fi

        for dataset_path in "${dataset_dirs[@]}"; do
            [ -d "$dataset_path" ] || continue
            dataset=$(basename "$dataset_path")
            if [[ ! " ${DATASETS[*]} " =~ " ${dataset} " ]]; then
                echo "Dataset $dataset not in DATASETS list, skipping."
                continue
            fi
            mapping_csv="${dataset_path}/${MAPPING_CSV_NAME}"

            if [ ! -f "$mapping_csv" ]; then
                echo "Mapping not found: $mapping_csv, skipping dataset $dataset."
                continue
            fi

            vad_dir="${dataset_path}/${VAD_DIR_NAME}"
            vad_jsonl="${vad_dir}/${VAD_JSONL_NAME}"
            mkdir -p "$vad_dir"

            echo "Running FireRedVAD for dataset: $dataset"
            echo "  mapping: $mapping_csv"
            echo "  output : $vad_jsonl"

            python3 "$VAD_SCRIPT" \
                --audio_mapping "$mapping_csv" \
                --output_jsonl "$vad_jsonl" \
                --model_dir "$FIRERED_VAD_MODEL_DIR" \
                --fireredasr2s_root "$FIREREDASR2S_ROOT" \
                --speech_threshold "$SPEECH_THRESHOLD" \
                --use_gpu "$USE_GPU"
        done
    done
    echo "VAD processing completed!"
}

run_timing_evaluation() {
    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Running timing evaluation for base directory: $BASE_DIR"
        BASE_NAME=$(basename "$BASE_DIR")
        RESULT_CSV="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.csv"
        RESULT_TXT="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.txt"

        # Phase-2a: prepare label_segments.jsonl (reads meta + overlap once per dataset)
        echo "Preparing label_segments.jsonl for each dataset..."
        python3 "$PREPARE_LABEL_SCRIPT" \
            --ground_truth_dir "$TEST_SET_DIR" \
            --metadata_dir "$METADATA_DIR" \
            --gt_json_base_dir "$GT_JSON_BASE_DIR" \
            --predicted_dir "$BASE_DIR" \
            --datasets "${DATASETS[*]}" \
            --vad_dir_name "$VAD_DIR_NAME" \
            --collar "$COLLAR" \
            --match_tolerance "$MATCH_TOLERANCE"

        # Phase-2b: eval from vad_segments.jsonl + label_segments.jsonl only (no meta/overlap)
        python3 "$EVAL_SCRIPT" \
            --ground_truth_dir "$TEST_SET_DIR" \
            --predicted_dir "$BASE_DIR" \
            --gt_json_base_dir "$GT_JSON_BASE_DIR" \
            --datasets "${DATASETS[*]}" \
            --vad_dir_name "$VAD_DIR_NAME" \
            --vad_jsonl_name "$VAD_JSONL_NAME" \
            --frame_shift "$FRAME_SHIFT" \
            --collar "$COLLAR" \
            --match_tolerance "$MATCH_TOLERANCE" \
            --save_path "$RESULT_CSV" \
            --report_path "$RESULT_TXT"

        echo "Timing evaluation completed:"
        echo "  detail : $RESULT_CSV"
        echo "  report : $RESULT_TXT"
    done
    echo "Timing evaluation completed:"
}

run_visualization() {
    for BASE_DIR in "${BASE_DIRS[@]}"; do
        echo "Running visualization for base directory: $BASE_DIR"
        
        # Note: We rely on Mode 2 (Timing Evaluation) to have already prepared
        # label_segments.jsonl with the necessary gender information.

        mapfile -t dataset_dirs < <(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d)
        
        for dataset_path in "${dataset_dirs[@]}"; do
            [ -d "$dataset_path" ] || continue
            dataset=$(basename "$dataset_path")
            if [[ ! " ${DATASETS[*]} " =~ " ${dataset} " ]]; then
                continue
            fi
            
            vad_dir="${dataset_path}/${VAD_DIR_NAME}"
            label_jsonl="${vad_dir}/label_segments.jsonl"
            vad_jsonl="${vad_dir}/${VAD_JSONL_NAME}"
            output_dir="${vad_dir}/figures"
            
            if [ ! -f "$label_jsonl" ]; then
                echo "Skipping $dataset: missing label segments ($label_jsonl)"
                echo "Please run Mode 2 first to prepare label segments."
                continue
            fi
            if [ ! -f "$vad_jsonl" ]; then
                echo "Skipping $dataset: missing VAD segments ($vad_jsonl)"
                continue
            fi
            
            mkdir -p "$output_dir"
            echo "Generating timeline figures for dataset: $dataset"
            
            # Construct path to metrics CSV
            # It should be at ${BASE_DIR}/${BASE_DIR_BASENAME}_TSE_TIMING.csv
            BASE_NAME=$(basename "$BASE_DIR")
            METRICS_CSV="${BASE_DIR}/${BASE_NAME}_TSE_TIMING.csv"
            
            if [ ! -f "$METRICS_CSV" ]; then
                echo "Warning: Metrics CSV not found at $METRICS_CSV. Figures will lack scores."
                python3 "$VIS_SCRIPT" \
                    --label_jsonl "$label_jsonl" \
                    --vad_jsonl "$vad_jsonl" \
                    --output_dir "$output_dir"
            else
                python3 "$VIS_SCRIPT" \
                    --label_jsonl "$label_jsonl" \
                    --vad_jsonl "$vad_jsonl" \
                    --output_dir "$output_dir" \
                    --metrics_csv "$METRICS_CSV"
            fi
        done
    done
    echo "Visualization completed!"
}

for mode in "${MODES[@]}"; do
    if [ "$mode" == "1" ]; then
        run_vad
    elif [ "$mode" == "2" ]; then
        run_timing_evaluation
    elif [ "$mode" == "3" ]; then
        run_visualization
    else
        echo "Invalid mode: $mode. Please use 1 (VAD), 2 (Timing Evaluation), or 3 (Visualization)."
        exit 1
    fi
done

echo "All tasks finished successfully!"
