#!/bin/bash

conda activate REAL-T && python3 ./utils/download_asr_model.py \
  --repo_id FireRedTeam/FireRedASR-AED-L \
  --save_dir ./FireRedASR/pretrained_models

mkdir -p ./datasets
conda activate REAL-T && python3 ./utils/download_REAL-T.py \
--save_dir "./datasets" \
--hub_repo "SLbaba/REAL-T"


MIX_DIR="./datasets/REAL-T/mixtures"
ENROL_DIR="./datasets/REAL-T/enrolment_speakers"
OUT_CSV="./datasets/REAL-T/mapping.csv"


echo "utterance,path" > "$OUT_CSV"

find "$MIX_DIR" -type f -name "*.wav" | while read -r file; do
    utt_id=$(basename "$file" .wav)
    abs_path=$(realpath "$file")
    echo "$utt_id,$abs_path" >> "$OUT_CSV"
done

find "$ENROL_DIR" -type f -name "*.wav" | while read -r file; do
    utt_id=$(basename "$file" .wav)
    abs_path=$(realpath "$file")
    echo "$utt_id,$abs_path" >> "$OUT_CSV"
done

echo "mapping.csv generated at $(realpath $OUT_CSV)"

