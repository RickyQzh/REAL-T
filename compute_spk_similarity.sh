#!/bin/bash

set -euo pipefail

conda activate RealT_py310

TEST_SET_DIR="${TEST_SET_DIR:-./datasets/REAL-T/PRIMARY}"
MAPPING_CSV="${MAPPING_CSV:-./datasets/REAL-T/mapping.csv}"
WESPEAKER_LANG="${WESPEAKER_LANG:-en}"
WESPEAKER_PROVIDER="${WESPEAKER_PROVIDER:-auto}"
WESPEAKER_DATASET_LANG_OVERRIDES="${WESPEAKER_DATASET_LANG_OVERRIDES:-AISHELL-4:chs,AliMeeting:chs}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SPK_SIM_PAIR_MODE="${SPK_SIM_PAIR_MODE:-tse_enrol}"

if [ -z "${BASE_DIRS:-}" ]; then
    BASE_DIRS="./output/PRIMARY/bsrnn_vox1"
fi

read -ra BASE_DIR_LIST <<< "$BASE_DIRS"
if [ "${#BASE_DIR_LIST[@]}" -eq 0 ]; then
    echo "No BASE_DIRS provided."
    exit 1
fi

if [ "$SPK_SIM_PAIR_MODE" != "tse_enrol" ] && [ "$SPK_SIM_PAIR_MODE" != "mixture_enrol" ]; then
    echo "Invalid SPK_SIM_PAIR_MODE=$SPK_SIM_PAIR_MODE (must be tse_enrol or mixture_enrol)."
    exit 1
fi

# Usage: 1 = compute & generate CSV only
#        2 = generate TXT from existing CSV (no WeSpeaker needed)
#        bash -i ./compute_spk_similarity.sh 1
#        bash -i ./compute_spk_similarity.sh 2
#        bash -i ./compute_spk_similarity.sh 1 2

MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    echo "No mode selected. Please specify 1 (compute & generate CSV), 2 (generate TXT from CSV), or both."
    exit 1
fi

run_spk_sim_full() {
    python3 - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("wespeakerruntime") is None:
    print("Dependency missing: wespeakerruntime. Install with: pip install wespeakerruntime")
    sys.exit(1)
PY

    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        CMD=(
            python3 ./utils/spk_similarity_eval.py
            --base_dir "$BASE_DIR"
            --test_set_dir "$TEST_SET_DIR"
            --mapping_csv "$MAPPING_CSV"
            --wespeaker_lang "$WESPEAKER_LANG"
            --provider "$WESPEAKER_PROVIDER"
            --dataset_lang_overrides "$WESPEAKER_DATASET_LANG_OVERRIDES"
            --num_workers "$NUM_WORKERS"
            --pair_mode "$SPK_SIM_PAIR_MODE"
            --csv_only
        )

        if [ -n "$MAX_SAMPLES" ]; then
            CMD+=(--max_samples "$MAX_SAMPLES")
        fi

        echo "Running spk-sim (mode 1: compute & CSV) for BASE_DIR=$BASE_DIR pair_mode=$SPK_SIM_PAIR_MODE"
        "${CMD[@]}"
    done
}

run_spk_sim_regen_txt() {
    for BASE_DIR in "${BASE_DIR_LIST[@]}"; do
        if [ ! -d "$BASE_DIR" ]; then
            echo "[Skip] Base directory does not exist: $BASE_DIR"
            continue
        fi

        CMD=(
            python3 ./utils/spk_similarity_eval.py
            --base_dir "$BASE_DIR"
            --wespeaker_lang "$WESPEAKER_LANG"
            --dataset_lang_overrides "$WESPEAKER_DATASET_LANG_OVERRIDES"
            --num_workers "$NUM_WORKERS"
            --pair_mode "$SPK_SIM_PAIR_MODE"
            --regen_txt_only
        )

        echo "Running spk-sim (mode 2: generate TXT) for BASE_DIR=$BASE_DIR pair_mode=$SPK_SIM_PAIR_MODE"
        "${CMD[@]}"
    done
}

for mode in "${MODES[@]}"; do
    if [ "$mode" == "1" ]; then
        run_spk_sim_full
    elif [ "$mode" == "2" ]; then
        run_spk_sim_regen_txt
    else
        echo "Invalid mode: $mode. Please use 1 (compute & CSV), 2 (generate TXT), or both."
        exit 1
    fi
done

echo "Speaker similarity evaluation finished."
