#!/usr/bin/env python3
"""Build enrol inventory with dataset and language evidence.

This script creates an inventory for all files under:
`datasets/REAL-T/enrolment_speakers`.
It resolves each enrol utterance to:
1) dataset source from `datasets/REAL-T/mapping/*_mixture_and_enrolment.csv`
2) language from `datasets/REAL-T/metadata/*_meta.csv` (`language` column)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "datasets" / "REAL-T"
    parser = argparse.ArgumentParser(
        description="Prepare enrol inventory CSV with dataset and ch/en labels."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=dataset_root,
        help="REAL-T dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "enrol_inventory.csv",
        help="Output inventory CSV path (default: dashboard/enrol_quality/enrol_inventory.csv)",
    )
    return parser.parse_args()


def load_dataset_mapping(mapping_dir: Path) -> dict[str, str]:
    enrol_to_dataset: dict[str, str] = {}
    for csv_path in sorted(mapping_dir.glob("*_mixture_and_enrolment.csv")):
        dataset = csv_path.name.replace("_mixture_and_enrolment.csv", "")
        df = pd.read_csv(csv_path)
        if "utterance" not in df.columns:
            raise ValueError(f"Missing 'utterance' column in {csv_path}")
        for enrol_id in df["utterance"].dropna().astype(str):
            if enrol_id in enrol_to_dataset and enrol_to_dataset[enrol_id] != dataset:
                raise ValueError(
                    f"Enrol '{enrol_id}' maps to multiple datasets: "
                    f"{enrol_to_dataset[enrol_id]} vs {dataset}"
                )
            enrol_to_dataset[enrol_id] = dataset
    return enrol_to_dataset


def load_dataset_language(metadata_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    dataset_to_lang_raw: dict[str, str] = {}
    dataset_to_evidence: dict[str, str] = {}
    for meta_csv in sorted(metadata_dir.glob("*_meta.csv")):
        dataset = meta_csv.name.replace("_meta.csv", "")
        df = pd.read_csv(meta_csv)
        if "language" not in df.columns:
            continue
        langs = sorted({str(x).strip() for x in df["language"].dropna().tolist() if str(x).strip()})
        if not langs:
            continue
        if len(langs) != 1:
            raise ValueError(f"Expected one language in {meta_csv}, got: {langs}")
        dataset_to_lang_raw[dataset] = langs[0]
        dataset_to_evidence[dataset] = f"{meta_csv}::language"
    return dataset_to_lang_raw, dataset_to_evidence


def normalize_ch_en(lang_raw: str) -> str:
    lang = lang_raw.lower()
    if lang in {"zh", "ch", "cn", "chinese"}:
        return "ch"
    if lang in {"en", "english"}:
        return "en"
    raise ValueError(f"Unsupported language tag: {lang_raw}")


def main() -> None:
    args = get_args()

    dataset_root = args.dataset_root.resolve()
    enrol_dir = dataset_root / "enrolment_speakers"
    mapping_dir = dataset_root / "mapping"
    metadata_dir = dataset_root / "metadata"

    if not enrol_dir.is_dir():
        raise FileNotFoundError(f"enrolment_speakers not found: {enrol_dir}")
    if not mapping_dir.is_dir():
        raise FileNotFoundError(f"mapping dir not found: {mapping_dir}")
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"metadata dir not found: {metadata_dir}")

    enrol_to_dataset = load_dataset_mapping(mapping_dir)
    dataset_to_lang_raw, dataset_to_evidence = load_dataset_language(metadata_dir)

    rows: list[dict[str, str]] = []
    wav_paths = sorted(enrol_dir.glob("*.wav"))
    for wav_path in wav_paths:
        enrol_id = wav_path.stem
        dataset = enrol_to_dataset.get(enrol_id)
        if dataset is None:
            raise KeyError(f"Cannot find dataset mapping for enrol_id: {enrol_id}")
        lang_raw = dataset_to_lang_raw.get(dataset)
        if lang_raw is None:
            raise KeyError(
                f"Cannot find language evidence for dataset '{dataset}' "
                f"(expected metadata/{dataset}_meta.csv with language column)."
            )
        rows.append(
            {
                "enrol_id": enrol_id,
                "dataset": dataset,
                "ch_en": normalize_ch_en(lang_raw),
                "audio_path": str(wav_path.resolve()),
                "language_evidence": dataset_to_evidence[dataset],
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["dataset", "enrol_id"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    summary = out_df.groupby(["dataset", "ch_en"]).size().reset_index(name="count")
    print(f"Saved inventory: {args.output_csv.resolve()} (rows={len(out_df)})")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
