#!/usr/bin/env python3
"""Merge shard transcript CSVs into one final enrol transcript CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Merge enrol ASR shard outputs.")
    parser.add_argument(
        "--model",
        choices=["whisper", "firered"],
        required=True,
        help="Model name used in shard filenames.",
    )
    parser.add_argument(
        "--shards_dir",
        type=Path,
        default=root / "shards",
        help="Directory containing shard csv files.",
    )
    parser.add_argument(
        "--inventory_csv",
        type=Path,
        default=root / "enrol_inventory.csv",
        help="Inventory CSV for coverage check.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output merged CSV. Default: dashboard/enrol_quality/enrol_transcripts_<model>.csv",
    )
    parser.add_argument(
        "--fill_missing_from_inventory",
        type=int,
        default=1,
        help="If 1, ensure all enrol_id in inventory appear in merged output (missing transcripts filled as empty).",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    shards_dir = args.shards_dir.resolve()
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {shards_dir}")

    pattern = f"{args.model}_shard*_of_*.csv"
    shard_files = sorted(shards_dir.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shards_dir} with pattern {pattern}")

    frames = []
    for csv_path in shard_files:
        df = pd.read_csv(csv_path)
        required = {"enrol_id", "dataset", "ch_en", "transcript"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["transcript"] = merged["transcript"].fillna("")
    merged["_transcript_len"] = merged["transcript"].astype(str).str.len()
    merged = (
        merged.sort_values(["enrol_id", "_transcript_len"], ascending=[True, False])
        .drop_duplicates(subset=["enrol_id"], keep="first")
        .drop(columns=["_transcript_len"])
    )

    final_cols = ["enrol_id", "dataset", "ch_en", "transcript"]
    merged = merged[final_cols].copy()

    expected = None
    missing_ids: list[str] = []
    if args.inventory_csv.is_file():
        inventory = pd.read_csv(args.inventory_csv)
        inventory = inventory[["enrol_id", "dataset", "ch_en"]].copy()
        inventory["enrol_id"] = inventory["enrol_id"].astype(str)
        merged["enrol_id"] = merged["enrol_id"].astype(str)
        expected = set(inventory["enrol_id"].dropna().astype(str))
        got = set(merged["enrol_id"].dropna().astype(str))
        missing_ids = sorted(expected - got)
        if args.fill_missing_from_inventory:
            merged = inventory.merge(
                merged[["enrol_id", "transcript"]],
                on="enrol_id",
                how="left",
            )
            merged["transcript"] = merged["transcript"].fillna("")

    merged = merged[final_cols].sort_values("enrol_id").reset_index(drop=True)

    if args.output_csv is None:
        output_csv = Path(__file__).resolve().parent / f"enrol_transcripts_{args.model}.csv"
    else:
        output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    if expected is not None:
        got = set(merged["enrol_id"].dropna().astype(str))
        missing_after_fill = sorted(expected - got)
        print(
            f"Saved merged CSV: {output_csv.resolve()} "
            f"(rows={len(merged)}, expected={len(expected)}, missing_before_fill={len(missing_ids)}, "
            f"missing_after_fill={len(missing_after_fill)})"
        )
        if missing_ids:
            print("Missing enrol_id sample:", ", ".join(missing_ids[:10]))
    else:
        print(f"Saved merged CSV: {output_csv.resolve()} (rows={len(merged)})")


if __name__ == "__main__":
    main()
