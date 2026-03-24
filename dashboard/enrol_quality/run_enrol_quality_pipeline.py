#!/usr/bin/env python3
"""Run the enrol-quality pipeline end to end and clean intermediates by default."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dashboard/enrol_quality/enrol_ter_full.csv from scratch. "
            "Intermediate inventory/ASR/GT files are written to a temporary work dir "
            "and removed automatically unless --keep_intermediates=1."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=REPO_ROOT / "datasets" / "REAL-T",
        help="REAL-T dataset root.",
    )
    parser.add_argument(
        "--json_root",
        type=Path,
        default=REPO_ROOT / "datasets" / "REAL-T" / "json",
        help="Root containing */no_overlap_segments.json.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=SCRIPT_DIR / "enrol_ter_full.csv",
        help="Final enrol TER report path.",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Optional working directory for temporary files.",
    )
    parser.add_argument(
        "--keep_intermediates",
        type=int,
        default=0,
        help="If 1, keep the temporary work directory for debugging.",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke helper scripts.",
    )
    parser.add_argument(
        "--ch_model",
        choices=["whisper", "firered"],
        default="firered",
        help="ASR backend for Chinese enrol utterances.",
    )
    parser.add_argument(
        "--en_model",
        choices=["whisper", "firered"],
        default="whisper",
        help="ASR backend for English enrol utterances.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Default device used by both ASR jobs unless overridden.",
    )
    parser.add_argument(
        "--ch_device",
        type=str,
        default=None,
        help="Optional device override for the Chinese ASR job.",
    )
    parser.add_argument(
        "--en_device",
        type=str,
        default=None,
        help="Optional device override for the English ASR job.",
    )
    parser.add_argument(
        "--whisper_model_name",
        type=str,
        default="openai/whisper-large-v2",
        help="Whisper model id used whenever a whisper job is selected.",
    )
    parser.add_argument(
        "--whisper_model_path",
        type=Path,
        default=None,
        help="Optional local whisper model path.",
    )
    parser.add_argument(
        "--auto_fallback_cpu",
        type=int,
        default=1,
        help="Forwarded to transcribe_enrol_asr.py.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests. Applies before language splitting.",
    )
    return parser.parse_args()


def _print_cmd(cmd: list[str]) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))


def _run(cmd: list[str]) -> None:
    _print_cmd(cmd)
    subprocess.run(cmd, check=True)


def _resolve_work_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, False
    temp_dir = tempfile.mkdtemp(prefix=".tmp_enrol_quality_", dir=str(SCRIPT_DIR))
    return Path(temp_dir).resolve(), True


def _maybe_trim_inventory(inventory_csv: Path, max_samples: int | None) -> Path:
    if max_samples is None:
        return inventory_csv
    df = pd.read_csv(inventory_csv)
    trimmed = df.head(max_samples).copy()
    trimmed_csv = inventory_csv.parent / f"{inventory_csv.stem}_head{max_samples}.csv"
    trimmed.to_csv(trimmed_csv, index=False)
    print(
        f"Trimmed inventory for smoke test: {trimmed_csv} "
        f"(rows={len(trimmed)} of original {len(df)})"
    )
    return trimmed_csv


def _transcribe_cmd(
    args: argparse.Namespace,
    inventory_csv: Path,
    language: str,
    model_name: str,
    device: str,
    output_csv: Path,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(SCRIPT_DIR / "transcribe_enrol_asr.py"),
        "--model",
        model_name,
        "--inventory_csv",
        str(inventory_csv),
        "--language",
        language,
        "--output_csv",
        str(output_csv),
        "--num_shards",
        "1",
        "--shard_index",
        "0",
        "--device",
        device,
        "--auto_fallback_cpu",
        str(args.auto_fallback_cpu),
    ]
    if model_name == "whisper":
        cmd.extend(["--whisper_model_name", args.whisper_model_name])
        if args.whisper_model_path is not None:
            cmd.extend(["--whisper_model_path", str(args.whisper_model_path.resolve())])
    return cmd


def _write_empty_asr_csv(path: Path) -> None:
    pd.DataFrame(columns=["enrol_id", "dataset", "ch_en", "transcript", "error"]).to_csv(
        path,
        index=False,
    )
    print(f"Wrote empty ASR CSV: {path}")


def main() -> None:
    args = get_args()
    dataset_root = args.dataset_root.resolve()
    json_root = args.json_root.resolve()
    output_csv = args.output_csv.resolve()
    keep_intermediates = bool(args.keep_intermediates)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not json_root.is_dir():
        raise FileNotFoundError(f"json_root not found: {json_root}")

    work_dir, auto_created = _resolve_work_dir(args)
    inventory_csv = work_dir / "enrol_inventory.csv"
    gt_csv = work_dir / "enrol_ground_truth_from_no_overlap.csv"
    ch_asr_csv = work_dir / "ch_asr.csv"
    en_asr_csv = work_dir / "en_asr.csv"

    try:
        _run(
            [
                args.python_executable,
                str(SCRIPT_DIR / "prepare_enrol_inventory.py"),
                "--dataset_root",
                str(dataset_root),
                "--output_csv",
                str(inventory_csv),
            ]
        )
        active_inventory_csv = _maybe_trim_inventory(inventory_csv, args.max_samples)
        active_inventory_df = pd.read_csv(active_inventory_csv)
        active_langs = {
            str(value).strip().lower()
            for value in active_inventory_df["ch_en"].dropna().tolist()
            if str(value).strip()
        }

        _run(
            [
                args.python_executable,
                str(SCRIPT_DIR / "extract_enrol_ground_truth.py"),
                "--inventory_csv",
                str(active_inventory_csv),
                "--json_root",
                str(json_root),
                "--output_csv",
                str(gt_csv),
            ]
        )

        ch_device = args.ch_device or args.device
        en_device = args.en_device or args.device
        if "ch" in active_langs:
            _run(
                _transcribe_cmd(
                    args=args,
                    inventory_csv=active_inventory_csv,
                    language="ch",
                    model_name=args.ch_model,
                    device=ch_device,
                    output_csv=ch_asr_csv,
                )
            )
        else:
            _write_empty_asr_csv(ch_asr_csv)

        if "en" in active_langs:
            _run(
                _transcribe_cmd(
                    args=args,
                    inventory_csv=active_inventory_csv,
                    language="en",
                    model_name=args.en_model,
                    device=en_device,
                    output_csv=en_asr_csv,
                )
            )
        else:
            _write_empty_asr_csv(en_asr_csv)

        _run(
            [
                args.python_executable,
                str(SCRIPT_DIR / "build_enrol_ter_report.py"),
                "--gt_csv",
                str(gt_csv),
                "--ch_asr_csv",
                str(ch_asr_csv),
                "--en_asr_csv",
                str(en_asr_csv),
                "--output_csv",
                str(output_csv),
            ]
        )
    finally:
        should_cleanup = not keep_intermediates
        if should_cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
            cleanup_note = "cleaned"
        else:
            cleanup_note = "kept"
        print(f"Intermediate work dir {cleanup_note}: {work_dir}")
        if auto_created and should_cleanup:
            print("Temporary intermediates were removed after the final CSV was produced.")

    print(f"Finished enrol-quality pipeline. Final report: {output_csv}")


if __name__ == "__main__":
    main()
