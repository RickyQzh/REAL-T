#!/bin/bash

set -euo pipefail

conda activate RealT_py310
# So that conda-installed libcudnn (e.g. libcudnn.so.9) is found when using CUDA provider
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T/PRIMARY}"
DNSMOS_MODEL_DIR="${DNSMOS_MODEL_DIR:-./DNSMOS}"
DNSMOS_PROVIDER="${DNSMOS_PROVIDER:-auto}"
DNSMOS_NO_DOWNLOAD="${DNSMOS_NO_DOWNLOAD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [ -z "${BASE_DIRS:-}" ]; then
    BASE_DIRS="./output/PRIMARY/bsrnn_vox1 ./output/PRIMARY/BSRNN"
fi

read -ra BASE_DIR_LIST <<< "$BASE_DIRS"
if [ "${#BASE_DIR_LIST[@]}" -eq 0 ]; then
    echo "No BASE_DIRS provided."
    exit 1
fi

# Usage: 1 = compute DNSMOS & generate CSV (and optionally TXT)
#        2 = generate TXT from existing CSV (no DNSMOS model needed)
#        bash -i ./compute_dnsmos.sh 1
#        bash -i ./compute_dnsmos.sh 2
#        bash -i ./compute_dnsmos.sh 1 2

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (compute & CSV), 2 (generate TXT from CSV), or both."
    exit 1
fi

run_dnsmos_full() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        CMD=(
            python3 ./utils/dnsmos_eval.py
            --base_dir "$BASE_DIR"
            --test_set_dir "$TEST_SET_DIR"
            --dnsmos_model_dir "$DNSMOS_MODEL_DIR"
            --provider "$DNSMOS_PROVIDER"
            --csv_only
        )
        if [ "${DNSMOS_NO_DOWNLOAD}" = "1" ]; then
            CMD+=(--no_download_models)
        fi
        if [ -n "$MAX_SAMPLES" ]; then
            CMD+=(--max_samples "$MAX_SAMPLES")
        fi

        echo "Running DNSMOS (mode 1: compute & CSV) for BASE_DIR=$BASE_DIR provider=$DNSMOS_PROVIDER"
        "${CMD[@]}"
    done
}

run_dnsmos_regen_txt() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        CMD=(
            python3 ./utils/dnsmos_eval.py
            --base_dir "$BASE_DIR"
            --dnsmos_model_dir "$DNSMOS_MODEL_DIR"
            --regen_txt_only
        )

        echo "Running DNSMOS (mode 2: generate TXT) for BASE_DIR=$BASE_DIR"
        "${CMD[@]}"
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" == "1" ]; then
        run_dnsmos_full
    elif [ "$mode" == "2" ]; then
        run_dnsmos_regen_txt
    else
        echo "Invalid mode: $mode. Please use 1 (compute & CSV), 2 (generate TXT), or both."
        exit 1
    fi
done

echo "DNSMOS evaluation finished."
