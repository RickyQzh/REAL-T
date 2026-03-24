#!/usr/bin/env python3
"""Transcribe REAL-T enrolment speakers with Whisper or FireRed ASR.

# Usage examples (single GPU, full enrol set):
# 1) Prepare inventory first:
#    python dashboard/enrol_quality/prepare_enrol_inventory.py
#
# 2) Whisper (single card full run):
#    CUDA_VISIBLE_DEVICES=0 python dashboard/enrol_quality/transcribe_enrol_asr.py \
#      --model whisper \
#      --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
#      --num_shards 1 --shard_index 0 \
#      --device cuda:0
#
#    (Quick smoke test on CPU, faster model):
#    python dashboard/enrol_quality/transcribe_enrol_asr.py \
#      --model whisper \
#      --whisper_model_name openai/whisper-tiny \
#      --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
#      --num_shards 1 --shard_index 0 \
#      --device cpu --max_samples 10
#
# 3) FireRedASR-AED-L (single card full run):
#    CUDA_VISIBLE_DEVICES=0 python dashboard/enrol_quality/transcribe_enrol_asr.py \
#      --model firered \
#      --inventory_csv dashboard/enrol_quality/enrol_inventory.csv \
#      --num_shards 1 --shard_index 0 \
#      --device cuda:0
#
# 4) Merge shard CSVs:
#    python dashboard/enrol_quality/merge_enrol_transcripts.py --model whisper
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ASR_DIR = REPO_ROOT / "asr"
if str(ASR_DIR) not in sys.path:
    sys.path.insert(0, str(ASR_DIR))

from asr_models import FireRedASR_AED_L_ASRModel, WhisperASR  # noqa: E402


def get_args() -> argparse.Namespace:
    default_inventory = Path(__file__).resolve().parent / "enrol_inventory.csv"
    default_shards_dir = Path(__file__).resolve().parent / "shards"
    parser = argparse.ArgumentParser(description="Transcribe enrolment speakers by ASR.")
    parser.add_argument(
        "--model",
        choices=["whisper", "firered"],
        required=True,
        help="ASR backend to use.",
    )
    parser.add_argument(
        "--inventory_csv",
        type=Path,
        default=default_inventory,
        help="Inventory CSV from prepare_enrol_inventory.py",
    )
    parser.add_argument(
        "--language",
        choices=["ch", "en"],
        default=None,
        help="Optional ch/en filter before sharding.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, write to shards/<model>_shardXXX_of_YYY.csv",
    )
    parser.add_argument(
        "--shards_dir",
        type=Path,
        default=default_shards_dir,
        help="Directory for shard outputs.",
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Total shard count.")
    parser.add_argument("--shard_index", type=int, default=0, help="Current shard index, 0-based.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--whisper_model_name",
        type=str,
        default="openai/whisper-large-v2",
        help="Whisper model id for --model whisper, e.g. openai/whisper-large-v2 or openai/whisper-tiny.",
    )
    parser.add_argument(
        "--whisper_model_path",
        type=Path,
        default=None,
        help="Optional local whisper model path. If set, overrides --whisper_model_name.",
    )
    parser.add_argument(
        "--auto_fallback_cpu",
        type=int,
        default=1,
        help="If CUDA kernel is incompatible at runtime, fallback to CPU automatically.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def to_model_language(ch_en: str) -> str:
    value = str(ch_en).strip().lower()
    if value == "ch":
        return "zh"
    if value == "en":
        return "en"
    raise ValueError(f"Unsupported ch_en value: {ch_en}")


def build_model(
    model: str,
    device: str,
    whisper_model_name: str = "openai/whisper-large-v2",
    whisper_model_path: Path | None = None,
):
    if model == "whisper":
        return WhisperASR(
            model_name=whisper_model_name,
            model_path=str(whisper_model_path) if whisper_model_path else None,
            device=device,
        )
    if model == "firered":
        return FireRedASR_AED_L_ASRModel(model_name="aed", device=device)
    raise ValueError(f"Unsupported model: {model}")


def slice_shard(df: pd.DataFrame, num_shards: int, shard_index: int) -> pd.DataFrame:
    if num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    per_shard = math.ceil(len(df) / num_shards)
    start = shard_index * per_shard
    end = min(start + per_shard, len(df))
    return df.iloc[start:end].reset_index(drop=True)


def main() -> None:
    args = get_args()

    inv_path = args.inventory_csv.resolve()
    if not inv_path.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inv_path}")

    df = pd.read_csv(inv_path)
    required_cols = {"enrol_id", "dataset", "ch_en", "audio_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in inventory CSV: {sorted(missing)}")
    df["ch_en"] = df["ch_en"].astype(str).str.strip().str.lower()

    if args.language is not None:
        df = df[df["ch_en"] == args.language].copy()
        if df.empty:
            raise ValueError(
                f"No rows matched --language={args.language} in inventory: {inv_path}"
            )

    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    shard_df = slice_shard(df, args.num_shards, args.shard_index)
    if shard_df.empty:
        raise ValueError("Selected shard is empty. Adjust --num_shards/--shard_index.")

    if args.output_csv is None:
        args.shards_dir.mkdir(parents=True, exist_ok=True)
        output_csv = (
            args.shards_dir
            / f"{args.model}_shard{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
        )
    else:
        output_csv = args.output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded inventory "
        f"{inv_path} (rows={len(df)}, language={args.language or 'all'}, "
        f"shard={args.shard_index + 1}/{args.num_shards}, shard_rows={len(shard_df)})"
    )

    model = build_model(
        args.model,
        args.device,
        whisper_model_name=args.whisper_model_name,
        whisper_model_path=args.whisper_model_path,
    )
    fallback_to_cpu_used = False

    records: list[dict[str, str]] = []
    for row in tqdm(shard_df.itertuples(index=False), total=len(shard_df), desc=f"ASR-{args.model}"):
        audio_path = Path(row.audio_path)
        if not audio_path.is_file():
            records.append(
                {
                    "enrol_id": row.enrol_id,
                    "dataset": row.dataset,
                    "ch_en": row.ch_en,
                    "transcript": "",
                    "error": f"Missing audio file: {audio_path}",
                }
            )
            continue

        lang = to_model_language(row.ch_en)
        try:
            transcript = model.transcribe_audio(str(audio_path), language=lang)
            records.append(
                {
                    "enrol_id": row.enrol_id,
                    "dataset": row.dataset,
                    "ch_en": row.ch_en,
                    "transcript": transcript,
                    "error": "",
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            err_text = f"{type(exc).__name__}: {exc}"
            can_fallback = (
                bool(args.auto_fallback_cpu)
                and not fallback_to_cpu_used
                and str(args.device).startswith("cuda")
                and "no kernel image is available for execution on the device"
                in str(exc).lower()
            )
            if can_fallback:
                print(
                    "Detected CUDA kernel incompatibility; switching ASR model to CPU "
                    "and retrying from current sample."
                )
                model = build_model(
                    args.model,
                    "cpu",
                    whisper_model_name=args.whisper_model_name,
                    whisper_model_path=args.whisper_model_path,
                )
                fallback_to_cpu_used = True
                try:
                    transcript = model.transcribe_audio(str(audio_path), language=lang)
                    records.append(
                        {
                            "enrol_id": row.enrol_id,
                            "dataset": row.dataset,
                            "ch_en": row.ch_en,
                            "transcript": transcript,
                            "error": "",
                        }
                    )
                    continue
                except Exception as retry_exc:  # pylint: disable=broad-except
                    err_text = f"{type(retry_exc).__name__}: {retry_exc}"

            records.append(
                {
                    "enrol_id": row.enrol_id,
                    "dataset": row.dataset,
                    "ch_en": row.ch_en,
                    "transcript": "",
                    "error": err_text,
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)

    failures = int((out_df["error"].fillna("") != "").sum())
    print(
        f"Saved shard transcription: {output_csv.resolve()} "
        f"(rows={len(out_df)}, failures={failures})"
    )


if __name__ == "__main__":
    main()
