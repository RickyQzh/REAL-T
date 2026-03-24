#!/usr/bin/env python3
"""Build enrol-level TER report from mixed ASR outputs and enrol GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from meeteval.wer.wer.siso import siso_word_error_rate
except Exception:  # pylint: disable=broad-except
    siso_word_error_rate = None


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "utils"

import sys

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from asr_metrics import normalizer_for_transcript  # noqa: E402


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate enrol-level TER CSV from GT and ASR shard outputs."
    )
    parser.add_argument(
        "--gt_csv",
        type=Path,
        default=root / "enrol_ground_truth_from_no_overlap.csv",
        help="GT CSV from extract_enrol_ground_truth.py",
    )
    parser.add_argument(
        "--ch_asr_csv",
        type=Path,
        default=root / "shards" / "ch_firered.csv",
        help="Chinese ASR output CSV (firered).",
    )
    parser.add_argument(
        "--en_asr_csv",
        type=Path,
        default=root / "shards" / "en_whisper.csv",
        help="English ASR output CSV (whisper).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=root / "enrol_ter_full.csv",
        help="Output full TER report CSV.",
    )
    parser.add_argument(
        "--empty_pred_ter",
        type=float,
        default=1.0,
        help="TER value for empty ASR prediction when GT is non-empty.",
    )
    return parser.parse_args()


def _load_asr_csv(path: Path, expected_lang: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"ASR CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"enrol_id", "dataset", "ch_en", "transcript"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["transcript"] = df["transcript"].fillna("").astype(str)
    df["ch_en"] = df["ch_en"].astype(str)
    df = df[df["ch_en"] == expected_lang].copy()
    if df.empty:
        return df
    # Keep the longest non-empty transcript if duplicate enrol_id appears.
    df["_len"] = df["transcript"].str.len()
    df = (
        df.sort_values(["enrol_id", "_len"], ascending=[True, False])
        .drop_duplicates(subset=["enrol_id"], keep="first")
        .drop(columns=["_len"])
    )
    return df


def _compute_ter_row(gt: str, pred: str, lang: str, empty_pred_ter: float) -> float:
    gt = (gt or "").strip()
    pred = (pred or "").strip()
    if not gt and not pred:
        return 0.0
    if gt and not pred:
        return float(empty_pred_ter)
    if not gt:
        return 0.0
    gt_norm = normalizer_for_transcript(gt, "Ground Truth", lang)
    pred_norm = normalizer_for_transcript(pred, "Predicted", lang)
    if siso_word_error_rate is not None:
        return float(siso_word_error_rate(gt_norm, pred_norm).error_rate)
    return _fallback_error_rate(gt_norm, pred_norm)


def _fallback_error_rate(ref: str, hyp: str) -> float:
    """Token-level edit distance / len(ref_tokens), used when meeteval is unavailable."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ri = ref_tokens[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )
    return float(dp[n][m] / n)


def main() -> None:
    args = get_args()
    gt_csv = args.gt_csv.resolve()
    ch_asr_csv = args.ch_asr_csv.resolve()
    en_asr_csv = args.en_asr_csv.resolve()
    output_csv = args.output_csv.resolve()

    if not gt_csv.is_file():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")

    gt = pd.read_csv(gt_csv)
    required_gt = {"enrol_id", "dataset", "ch_en", "ground_truth_transcript"}
    missing_gt = required_gt - set(gt.columns)
    if missing_gt:
        raise ValueError(f"{gt_csv} missing columns: {sorted(missing_gt)}")
    gt = gt[["enrol_id", "dataset", "ch_en", "ground_truth_transcript"]].copy()
    gt["ground_truth_transcript"] = gt["ground_truth_transcript"].fillna("").astype(str)

    ch_df = _load_asr_csv(ch_asr_csv, expected_lang="ch")
    en_df = _load_asr_csv(en_asr_csv, expected_lang="en")
    asr = pd.concat([ch_df, en_df], ignore_index=True)
    asr = asr[["enrol_id", "transcript"]].copy()
    asr = asr.rename(columns={"transcript": "enrol_asr_result"})

    out = gt.merge(asr, on="enrol_id", how="left")
    out["enrol_asr_result"] = out["enrol_asr_result"].fillna("").astype(str)

    lang_map = {"ch": "zh", "en": "en"}
    out["TER"] = np.nan
    for idx, row in out.iterrows():
        ch_en = str(row["ch_en"]).strip().lower()
        if ch_en not in lang_map:
            raise ValueError(f"Unsupported ch_en value: {row['ch_en']}")
        out.at[idx, "TER"] = _compute_ter_row(
            gt=row["ground_truth_transcript"],
            pred=row["enrol_asr_result"],
            lang=lang_map[ch_en],
            empty_pred_ter=args.empty_pred_ter,
        )

    out = out[
        ["enrol_id", "dataset", "ch_en", "ground_truth_transcript", "enrol_asr_result", "TER"]
    ].sort_values(["dataset", "enrol_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    has_pred = (out["enrol_asr_result"].astype(str).str.strip() != "").sum()
    print(
        f"Saved: {output_csv} (rows={len(out)}, with_pred={int(has_pred)}, "
        f"without_pred={len(out)-int(has_pred)}, avg_TER={out['TER'].mean():.6f})"
    )
    by_lang = out.groupby("ch_en")["TER"].mean().reset_index()
    print(by_lang.to_string(index=False))


if __name__ == "__main__":
    main()
