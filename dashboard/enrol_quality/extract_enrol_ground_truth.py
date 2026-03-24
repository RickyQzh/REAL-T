#!/usr/bin/env python3
"""Extract per-enrol ground-truth transcript from no_overlap_segments.json.

The enrol id format is produced by REAL-T-Ext `generate_datasets/construct.py`:
    f"{utterance_id}_{speaker}_{start:.2f}_{end:.2f}"
This script rebuilds the same key from `datasets/REAL-T/json/*/no_overlap_segments.json`
and joins it with `dashboard/enrol_quality/enrol_inventory.csv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def get_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract enrol-level ground truth transcript from no_overlap JSON files."
    )
    parser.add_argument(
        "--inventory_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "enrol_inventory.csv",
        help="Inventory CSV from prepare_enrol_inventory.py",
    )
    parser.add_argument(
        "--json_root",
        type=Path,
        default=repo_root / "datasets" / "REAL-T" / "json",
        help="Root directory containing dataset subdirs with no_overlap_segments.json",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "enrol_ground_truth_from_no_overlap.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=1,
        help="If 1, fail when any enrol_id is missing from no_overlap JSON map.",
    )
    return parser.parse_args()


def build_enrol_gt_map(json_root: Path) -> dict[str, dict[str, object]]:
    enrol_map: dict[str, dict[str, object]] = {}
    json_files = sorted(json_root.glob("*/no_overlap_segments.json"))
    if not json_files:
        raise FileNotFoundError(f"No no_overlap_segments.json found under: {json_root}")

    for json_path in json_files:
        dataset = json_path.parent.name
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for utterance_id, segments in data.items():
            for seg in segments:
                speaker = str(seg["speaker"])
                start = float(seg["start_time"])
                end = float(seg["end_time"])
                transcript = str(seg.get("transcript", "")).strip()

                # Same naming rule as REAL-T-Ext construct.py
                enrol_id = f"{utterance_id}_{speaker}_{start:.2f}_{end:.2f}"

                row = {
                    "enrol_id": enrol_id,
                    "json_dataset": dataset,
                    "utterance_id": str(utterance_id),
                    "speaker": speaker,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "ground_truth_transcript": transcript,
                    "json_path": str(json_path.resolve()),
                }

                if enrol_id in enrol_map:
                    prev = enrol_map[enrol_id]
                    # Same enrol_id should be unique globally; check consistency.
                    if prev["ground_truth_transcript"] != transcript or prev["json_dataset"] != dataset:
                        raise ValueError(
                            f"Conflicting mapping for enrol_id={enrol_id}: "
                            f"{prev['json_dataset']} vs {dataset}, "
                            f"text='{prev['ground_truth_transcript']}' vs '{transcript}'"
                        )
                else:
                    enrol_map[enrol_id] = row

    return enrol_map


def main() -> None:
    args = get_args()
    inventory_csv = args.inventory_csv.resolve()
    json_root = args.json_root.resolve()
    output_csv = args.output_csv.resolve()

    if not inventory_csv.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory_csv}")
    if not json_root.is_dir():
        raise FileNotFoundError(f"json_root not found: {json_root}")

    inv = pd.read_csv(inventory_csv)
    required_cols = {"enrol_id", "dataset", "ch_en"}
    missing_cols = required_cols - set(inv.columns)
    if missing_cols:
        raise ValueError(f"Inventory missing required columns: {sorted(missing_cols)}")

    gt_map = build_enrol_gt_map(json_root)
    gt_df = pd.DataFrame(list(gt_map.values()))

    out = inv.merge(gt_df, on="enrol_id", how="left")
    out["ground_truth_transcript"] = out["ground_truth_transcript"].fillna("")

    missing_mask = out["ground_truth_transcript"].astype(str).str.strip() == ""
    missing_count = int(missing_mask.sum())
    if args.strict and missing_count > 0:
        missing_ids = out.loc[missing_mask, "enrol_id"].astype(str).tolist()
        raise RuntimeError(
            f"Missing ground truth for {missing_count} enrol_id from no_overlap JSON. "
            f"Sample: {missing_ids[:10]}"
        )

    # Keep dataset from inventory as canonical dataset label.
    out["dataset_match"] = (out["dataset"].astype(str) == out["json_dataset"].astype(str)).astype(int)

    final_cols = [
        "enrol_id",
        "dataset",
        "ch_en",
        "ground_truth_transcript",
        "utterance_id",
        "speaker",
        "start_time",
        "end_time",
        "json_dataset",
        "dataset_match",
        "json_path",
    ]
    out = out[final_cols].sort_values(["dataset", "enrol_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    dataset_stats = (
        out.assign(has_text=out["ground_truth_transcript"].astype(str).str.strip() != "")
        .groupby("dataset")["has_text"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "with_text"})
        .reset_index()
    )
    mismatch_count = int((out["dataset_match"] == 0).sum())
    print(
        f"Saved: {output_csv} (rows={len(out)}, missing_text={missing_count}, "
        f"dataset_mismatch={mismatch_count})"
    )
    print(dataset_stats.to_string(index=False))


if __name__ == "__main__":
    main()
