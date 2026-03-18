#!/bin/bash

set -euo pipefail

REAL_T_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REAL_T_ROOT}/env_setup.sh"

usage() {
    cat <<'EOF'
Usage:
  bash ./run_eval.sh --base-dir <path> --test-set PRIMARY|BASE --cuda <id> [--include-fisher]
EOF
}

BASE_DIR=""
TEST_SET=""
CUDA_ID=""
INCLUDING_FISHER_FLAG="False"

while [ $# -gt 0 ]; do
    case "$1" in
        --base-dir)
            BASE_DIR="${2:-}"
            shift 2
            ;;
        --test-set)
            TEST_SET="${2:-}"
            shift 2
            ;;
        --cuda)
            CUDA_ID="${2:-}"
            shift 2
            ;;
        --include-fisher)
            INCLUDING_FISHER_FLAG="True"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$BASE_DIR" ] || [ -z "$TEST_SET" ] || [ -z "$CUDA_ID" ]; then
    usage
    exit 1
fi

if [ "$TEST_SET" != "PRIMARY" ] && [ "$TEST_SET" != "BASE" ]; then
    echo "--test-set must be PRIMARY or BASE."
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Base directory not found: $BASE_DIR"
    exit 1
fi

TEST_SET_DIR="./datasets/REAL-T/${TEST_SET}"
if [ ! -d "$TEST_SET_DIR" ]; then
    echo "Test set directory not found: $TEST_SET_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_ID"
export BASE_DIRS="$BASE_DIR"
export TEST_SET_DIR
export INCLUDING_FISHER="$INCLUDING_FISHER_FLAG"
export USE_GPU=1
export ASR_DEVICE="cuda:0"
export WESPEAKER_PROVIDER="cuda"
export DNSMOS_PROVIDER="cuda"

echo "Running full eval pipeline"
echo "  base_dir : $BASE_DIR"
echo "  test_set : $TEST_SET"
echo "  cuda     : $CUDA_VISIBLE_DEVICES"
echo "  fisher   : $INCLUDING_FISHER"

run_stage() {
    local label="$1"
    shift
    echo
    echo "===== $label ====="
    "$@"
}

run_stage "TER" bash "${REAL_T_ROOT}/eval/transcribe_and_evaluation.sh" 1 2
run_stage "TER_ASR2_AED" bash "${REAL_T_ROOT}/eval/transcribe_and_evaluation_asr2.sh" 1 2
run_stage "TSE_TIMING" bash "${REAL_T_ROOT}/eval/vad_and_evaluation.sh" 1 2
run_stage "SPK_SIM_TSE_ENROL" bash "${REAL_T_ROOT}/eval/compute_spk_similarity.sh" 1 2
run_stage "SPK_SIM_MIXTURE_ENROL" env SPK_SIM_PAIR_MODE=mixture_enrol bash "${REAL_T_ROOT}/eval/compute_spk_similarity.sh" 1 2
run_stage "DNSMOS" bash "${REAL_T_ROOT}/eval/compute_dnsmos.sh" 1 2

echo
echo "Full eval pipeline completed successfully."
