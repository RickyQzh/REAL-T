from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_BASE_DIR = REPO_ROOT / "output" / "BASE"
OUTPUT_PRIMARY_DIR = REPO_ROOT / "output" / "PRIMARY"
METADATA_BASE_DIR = REPO_ROOT / "datasets" / "REAL-T" / "BASE"
METADATA_FULL_DIR = REPO_ROOT / "datasets" / "REAL-T" / "metadata"
DEFAULT_ENROL_TER_CSV = REPO_ROOT / "dashboard" / "enrol_quality" / "enrol_ter_full.csv"
ENROL_TER_CSV_ENV_KEY = "REALT_ENROL_TER_FULL_CSV"

RESULT_ROOT_BASE = "BASE"
RESULT_ROOT_PRIMARY = "PRIMARY"

# Virtual selectbox key: merges sim_enrol_mixture + sim_enrol_tse in the UI.
SIM_UI_KEY = "sim"
SIM_METRIC_KEYS = frozenset({"sim_enrol_mixture", "sim_enrol_tse"})
_SIM_PAIR_COLUMNS = [
    "model",
    "utterance_key",
    "dataset",
    "lang_display",
    "v_mix",
    "v_tse",
    "uplift_pct",
]

DATASET_ORDER = ["AISHELL-4", "AMI", "AliMeeting", "CHiME6", "DipCo"]
LANGUAGE_ORDER = ["en", "chs"]
SPEAKER_COUNT_OPTIONS = [2, 3, 4, 5]

FILTER_KEYS = [
    "selected_speakers",
    "speaker_scope",
    "speaker_ratio_min",
    "mixture_ratio_min",
    "mixture_duration_max",
    "transcript_length_min",
    "enrol_quality_max",
    "enrol_gt_length_filter",
]

PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "BASE": {
        "selected_speakers": SPEAKER_COUNT_OPTIONS.copy(),
        "speaker_scope": "all",
        "speaker_ratio_min": 20,
        "mixture_ratio_min": None,
        "mixture_duration_max": None,
        "transcript_length_min": 5,
        "enrol_quality_max": None,
        "enrol_gt_length_filter": "all",
    },
    "PRIMARY": {
        "selected_speakers": SPEAKER_COUNT_OPTIONS.copy(),
        "speaker_scope": "primary",
        "speaker_ratio_min": 20,
        "mixture_ratio_min": None,
        "mixture_duration_max": 30,
        "transcript_length_min": 5,
        "enrol_quality_max": None,
        "enrol_gt_length_filter": "all",
    },
}

METRIC_SPECS = {
    "ter_whisper": {
        "label": "TER / fireredasr-1/whisper",
        "file_suffix": "_TER.csv",
        "value_column": "wer_or_cer",
        "source_type": "ter",
    },
    "ter_fireredasr2": {
        "label": "TER / fireredasr-2",
        "file_suffix": "_TER_ASR2_AED.csv",
        "value_column": "wer_or_cer",
        "source_type": "ter",
    },
    "sim_enrol_mixture": {
        "label": "SIM / enrol-mixture",
        "file_suffix": "_spk_similarity_mixture_enrol.csv",
        "value_column": "speaker_cosine_similarity",
        "source_type": "utterance",
    },
    "sim_enrol_tse": {
        "label": "SIM / enrol-tse",
        "file_suffix": "_spk_similarity.csv",
        "value_column": "speaker_cosine_similarity",
        "source_type": "utterance",
    },
    "dnsmos_sig": {
        "label": "DNSMOS / SIG",
        "file_suffix": "_dnsmos.csv",
        "value_column": "SIG",
        "source_type": "utterance",
    },
    "dnsmos_bak": {
        "label": "DNSMOS / BAK",
        "file_suffix": "_dnsmos.csv",
        "value_column": "BAK",
        "source_type": "utterance",
    },
    "dnsmos_ovrl": {
        "label": "DNSMOS / OVRL",
        "file_suffix": "_dnsmos.csv",
        "value_column": "OVRL",
        "source_type": "utterance",
    },
    "dnsmos_p808": {
        "label": "DNSMOS / P808",
        "file_suffix": "_dnsmos.csv",
        "value_column": "P808",
        "source_type": "utterance",
    },
    "ratio_precision": {
        "label": "RATIO / precision",
        "file_suffix": "_TSE_TIMING.csv",
        "value_column": "precision",
        "source_type": "utterance",
    },
    "ratio_recall": {
        "label": "RATIO / recall",
        "file_suffix": "_TSE_TIMING.csv",
        "value_column": "recall",
        "source_type": "utterance",
    },
    "ratio_f1": {
        "label": "RATIO / f1",
        "file_suffix": "_TSE_TIMING.csv",
        "value_column": "f1",
        "source_type": "utterance",
    },
}

STATUS_LABELS = {
    "ok": "OK",
    "empty": "EMPTY",
    "missing": "MISSING",
    "error": "ERROR",
}


def is_sim_ui_metric(metric_key: str) -> bool:
    return metric_key == SIM_UI_KEY


def iter_selectable_metric_keys() -> list[str]:
    """Sidebar order: single SIM entry replaces the two underlying SIM metrics."""
    out: list[str] = []
    inserted_sim = False
    for key in METRIC_SPECS:
        if key in SIM_METRIC_KEYS:
            if not inserted_sim:
                out.append(SIM_UI_KEY)
                inserted_sim = True
            continue
        out.append(key)
    return out


def selectable_metric_label(metric_key: str) -> str:
    if metric_key == SIM_UI_KEY:
        return "SIM"
    return METRIC_SPECS[metric_key]["label"]


def normalize_sidebar_metric_key(metric_key: str) -> str:
    """Map legacy session values onto the unified SIM key."""
    if metric_key in SIM_METRIC_KEYS:
        return SIM_UI_KEY
    return metric_key


@dataclass(frozen=True)
class DashboardState:
    metadata_df: pd.DataFrame
    metric_long_df: pd.DataFrame
    availability_df: pd.DataFrame
    model_source_df: pd.DataFrame
    models: list[str]
    transcript_length_note: str | None
    enrol_quality_note: str | None
    enrol_quality_source_path: str | None


@dataclass(frozen=True)
class FilterConfig:
    preset_name: str
    selected_speakers: tuple[int, ...]
    speaker_scope: str
    speaker_ratio_min: int
    mixture_ratio_min: int | None
    mixture_duration_max: int | None
    transcript_length_min: int
    enrol_quality_max: float | None
    enrol_gt_length_filter: str


def normalize_language(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"zh", "zho", "chs", "ch", "cn", "chinese", "mandarin"}:
        return "chs"
    if text in {"en", "eng", "english"}:
        return "en"
    return text


def _empty_model_source_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["model", "base_dir", "primary_dir"])


def _empty_metric_long_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "lang_display",
            "utterance_key",
            "mixture_utterance",
            "enrolment_speakers_utterance",
            "total_number_of_speaker",
            "is_primary_speaker",
            "speaker_ratio",
            "mixture_ratio",
            "mixture_duration",
            "transcript_length",
            "enrol_ter",
            "enrol_gt_length",
            "is_official_primary",
            "metric_value",
            "model",
            "metric_key",
            "metric_label",
            "metric_available",
            "source_file",
            "result_root",
        ]
    )


def _empty_availability_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "metric_key",
            "metric_label",
            "result_root",
            "status",
            "status_label",
            "file_path",
            "row_count",
            "matched_row_count",
            "message",
        ]
    )


def _optional_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _result_root_dir(result_root: str) -> Path:
    if result_root == RESULT_ROOT_BASE:
        return OUTPUT_BASE_DIR
    if result_root == RESULT_ROOT_PRIMARY:
        return OUTPUT_PRIMARY_DIR
    raise ValueError(f"Unsupported result root: {result_root}")


def _build_model_source_df() -> pd.DataFrame:
    records: dict[str, dict[str, str | None]] = {}
    root_specs = [
        (RESULT_ROOT_BASE, OUTPUT_BASE_DIR, "base_dir"),
        (RESULT_ROOT_PRIMARY, OUTPUT_PRIMARY_DIR, "primary_dir"),
    ]
    for _result_root, root_dir, column_name in root_specs:
        if not root_dir.is_dir():
            continue
        for path in sorted(root_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir():
                continue
            record = records.setdefault(
                path.name,
                {
                    "model": path.name,
                    "base_dir": None,
                    "primary_dir": None,
                },
            )
            record[column_name] = str(path.resolve())

    if not records:
        return _empty_model_source_df()

    return pd.DataFrame.from_records(
        sorted(records.values(), key=lambda item: str(item["model"]).lower())
    )


def _resolve_effective_result_root_for_model(
    preset_name: str,
    base_dir: object,
    primary_dir: object,
) -> str | None:
    has_base = _optional_path(base_dir) is not None
    has_primary = _optional_path(primary_dir) is not None
    if preset_name == RESULT_ROOT_BASE:
        return RESULT_ROOT_BASE if has_base else None
    if preset_name == RESULT_ROOT_PRIMARY:
        if has_base:
            return RESULT_ROOT_BASE
        if has_primary:
            return RESULT_ROOT_PRIMARY
        return None
    raise ValueError(f"Unsupported preset_name: {preset_name}")


def _resolve_effective_result_root_lookup(
    model_source_df: pd.DataFrame,
    preset_name: str,
) -> dict[str, str | None]:
    if model_source_df.empty:
        return {}
    lookup: dict[str, str | None] = {}
    for row in model_source_df.to_dict(orient="records"):
        lookup[str(row["model"])] = _resolve_effective_result_root_for_model(
            preset_name=preset_name,
            base_dir=row.get("base_dir"),
            primary_dir=row.get("primary_dir"),
        )
    return lookup


def _expected_metric_csv_path(model: str, metric_key: str, result_root: str) -> Path:
    spec = METRIC_SPECS[metric_key]
    return _result_root_dir(result_root) / model / f"{model}{spec['file_suffix']}"


def make_utterance_key(mixture_utterance: object, enrolment_speakers_utterance: object) -> str:
    return f"{mixture_utterance}-{enrolment_speakers_utterance}"


def _fallback_normalize_for_length(transcript: object, language: object) -> str:
    text = "" if transcript is None or pd.isna(transcript) else str(transcript)
    lang = normalize_language(language)
    if lang == "chs":
        text = re.sub(r"\s+", "", text)
        return " ".join(text.strip())
    text = text.replace("…", "").replace(".", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_transcript_length_normalizer() -> tuple[Any, str | None]:
    try:
        from utils.asr_metrics import normalizer_for_transcript

        def compute_length(transcript: object, language: object) -> int:
            lang = normalize_language(language)
            lang = "zh" if lang == "chs" else lang
            normalized = normalizer_for_transcript(
                "" if transcript is None or pd.isna(transcript) else str(transcript),
                "Ground Truth",
                lang,
            )
            return len(normalized.strip().split()) if isinstance(normalized, str) else 0

        return compute_length, None
    except Exception as exc:  # pragma: no cover - fallback depends on env
        note = (
            "Falling back to a lightweight transcript-length normalizer because "
            f"`utils.asr_metrics.normalizer_for_transcript` is unavailable: {exc}"
        )

        def compute_length(transcript: object, language: object) -> int:
            normalized = _fallback_normalize_for_length(transcript, language)
            return len(normalized.strip().split()) if normalized else 0

        return compute_length, note


def _resolve_enrol_ter_csv_path() -> Path:
    env_path = os.getenv(ENROL_TER_CSV_ENV_KEY, "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_ENROL_TER_CSV.resolve()


@lru_cache(maxsize=1)
def _load_primary_annotations_from_full() -> pd.DataFrame:
    cols = ["utterance_key", "is_primary_speaker_from_full", "is_official_primary"]
    if not METADATA_FULL_DIR.is_dir():
        return pd.DataFrame(columns=cols)

    length_fn, _ = _get_transcript_length_normalizer()
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(METADATA_FULL_DIR.glob("*_meta.csv")):
        if csv_path.stem == "Fisher_meta":
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        work = df.copy()
        work["dataset"] = work["source"]
        work = work[work["dataset"].isin(DATASET_ORDER)].copy()
        work["utterance_key"] = work.apply(
            lambda row: make_utterance_key(
                row["mixture_utterance"], row["enrolment_speakers_utterance"]
            ),
            axis=1,
        )
        work["speaker_ratio"] = pd.to_numeric(work["speaker_ratio"], errors="coerce")
        work["mixture_duration"] = pd.to_numeric(work["mixture_duration"], errors="coerce")
        work["transcript_length"] = work.apply(
            lambda row: length_fn(row["ground_truth_transcript"], row["language"]), axis=1
        )
        frames.append(
            work[
                [
                    "utterance_key",
                    "mixture_utterance",
                    "speaker_ratio",
                    "mixture_duration",
                    "transcript_length",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(columns=cols)

    full_df = pd.concat(frames, ignore_index=True)
    max_ratio = full_df.groupby("mixture_utterance")["speaker_ratio"].transform("max")
    full_df["is_primary_speaker_from_full"] = full_df["speaker_ratio"].eq(max_ratio)
    full_df["is_official_primary"] = (
        full_df["is_primary_speaker_from_full"]
        & (full_df["speaker_ratio"] > 0.20)
        & (full_df["transcript_length"] > 5)
        & (full_df["mixture_duration"] <= 30)
    )
    return (
        full_df[cols]
        .groupby("utterance_key", as_index=False)
        .max()
        .sort_values("utterance_key")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _load_enrol_quality_annotations() -> tuple[pd.DataFrame, str | None, Path]:
    path = _resolve_enrol_ter_csv_path()
    cols = [
        "enrol_id",
        "enrol_ter",
        "enrol_gt_length",
        "enrol_quality_dataset",
        "enrol_quality_lang_display",
    ]
    if not path.is_file():
        note = (
            f"Enrol quality CSV not found at `{path}`. "
            "New filters `enrol quality` / `enrol length` will only be effective when this file exists."
        )
        return pd.DataFrame(columns=cols), note, path

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - depends on local file
        note = (
            f"Failed to read enrol quality CSV `{path}`: {exc}. "
            "New enrol-level filters are unavailable."
        )
        return pd.DataFrame(columns=cols), note, path

    required = {"enrol_id", "dataset", "ch_en", "ground_truth_transcript", "TER"}
    if not required.issubset(df.columns):
        note = (
            f"Enrol quality CSV `{path}` is missing columns: "
            f"{sorted(required.difference(df.columns))}. New enrol-level filters are unavailable."
        )
        return pd.DataFrame(columns=cols), note, path

    length_fn, length_note = _get_transcript_length_normalizer()

    work = df.copy()
    work["enrol_id"] = work["enrol_id"].astype(str)
    work["enrol_ter"] = pd.to_numeric(work["TER"], errors="coerce")
    work["enrol_quality_dataset"] = work["dataset"].astype(str)
    work["enrol_quality_lang_display"] = work["ch_en"].map(normalize_language)

    def _gt_len(row: pd.Series) -> int:
        lang = "zh" if row["enrol_quality_lang_display"] == "chs" else "en"
        return length_fn(row["ground_truth_transcript"], lang)

    work["enrol_gt_length"] = work.apply(_gt_len, axis=1)
    work = work[cols].copy()

    duplicated = int(work.duplicated(subset=["enrol_id"]).sum())
    if duplicated > 0:
        work = work.sort_values(["enrol_id"]).drop_duplicates(subset=["enrol_id"], keep="first")

    note_parts: list[str] = []
    if duplicated > 0:
        note_parts.append(
            f"Enrol quality CSV `{path}` has {duplicated} duplicate enrol_id rows; kept first occurrence."
        )
    if length_note:
        note_parts.append(length_note)
    note = " ".join(note_parts) if note_parts else None

    return work.reset_index(drop=True), note, path


@lru_cache(maxsize=1)
def load_base_metadata() -> tuple[pd.DataFrame, str | None, str | None, str | None]:
    length_fn, note = _get_transcript_length_normalizer()
    enrol_quality_df, enrol_quality_note, enrol_quality_path = _load_enrol_quality_annotations()
    primary_annotations_df = _load_primary_annotations_from_full()

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(METADATA_BASE_DIR.glob("*_meta.csv")):
        if csv_path.stem == "Fisher_meta":
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        work = df.copy()
        work["dataset"] = work["source"]
        work["lang_display"] = work["language"].map(normalize_language)
        work = work[work["dataset"].isin(DATASET_ORDER)].copy()
        work["utterance_key"] = work.apply(
            lambda row: make_utterance_key(
                row["mixture_utterance"], row["enrolment_speakers_utterance"]
            ),
            axis=1,
        )
        work["speaker_ratio"] = pd.to_numeric(work["speaker_ratio"], errors="coerce")
        work["mixture_ratio"] = pd.to_numeric(work["mixture_ratio"], errors="coerce")
        work["mixture_duration"] = pd.to_numeric(work["mixture_duration"], errors="coerce")
        work["total_number_of_speaker"] = pd.to_numeric(
            work["total_number_of_speaker"], errors="coerce"
        ).astype("Int64")
        work["transcript_length"] = work.apply(
            lambda row: length_fn(row["ground_truth_transcript"], row["language"]), axis=1
        )
        if not enrol_quality_df.empty:
            work = work.merge(
                enrol_quality_df,
                left_on="enrolment_speakers_utterance",
                right_on="enrol_id",
                how="left",
            )
            work["enrol_dataset_match"] = work["dataset"].eq(work["enrol_quality_dataset"])
        else:
            work["enrol_ter"] = np.nan
            work["enrol_gt_length"] = np.nan
        base_max_ratio = work.groupby("mixture_utterance")["speaker_ratio"].transform("max")
        work["is_primary_speaker_base"] = work["speaker_ratio"].eq(base_max_ratio)
        if not primary_annotations_df.empty:
            work = work.merge(primary_annotations_df, on="utterance_key", how="left")
            work["is_primary_speaker"] = (
                work["is_primary_speaker_from_full"]
                .fillna(work["is_primary_speaker_base"])
                .astype(bool)
            )
            work["is_official_primary"] = work["is_official_primary"].fillna(False).astype(bool)
        else:
            work["is_primary_speaker"] = work["is_primary_speaker_base"]
            work["is_official_primary"] = (
                work["is_primary_speaker"]
                & (work["speaker_ratio"] > 0.20)
                & (work["transcript_length"] > 5)
                & (work["mixture_duration"] <= 30)
            )
        frames.append(
            work[
                [
                    "dataset",
                    "source",
                    "language",
                    "lang_display",
                    "mixture_utterance",
                    "enrolment_speakers_utterance",
                    "utterance_key",
                    "speaker",
                    "gender",
                    "total_number_of_speaker",
                    "speaker_ratio",
                    "mixture_ratio",
                    "enrolment_speakers_duration",
                    "mixture_overlap_duration",
                    "mixture_duration",
                    "ground_truth_transcript",
                    "transcript_length",
                    "enrol_ter",
                    "enrol_gt_length",
                    "is_official_primary",
                    "is_primary_speaker",
                ]
            ]
        )

    if not frames:
        empty_df = pd.DataFrame(
            columns=[
                "dataset",
                "source",
                "language",
                "lang_display",
                "mixture_utterance",
                "enrolment_speakers_utterance",
                "utterance_key",
                "speaker",
                "gender",
                "total_number_of_speaker",
                "speaker_ratio",
                "mixture_ratio",
                "enrolment_speakers_duration",
                "mixture_overlap_duration",
                "mixture_duration",
                "ground_truth_transcript",
                "transcript_length",
                "enrol_ter",
                "enrol_gt_length",
                "is_official_primary",
                "is_primary_speaker",
            ]
        )
        return empty_df, note, enrol_quality_note, str(enrol_quality_path)

    metadata_df = pd.concat(frames, ignore_index=True)
    metadata_df = metadata_df.drop_duplicates(subset=["utterance_key"]).reset_index(drop=True)
    return metadata_df, note, enrol_quality_note, str(enrol_quality_path)


def _base_metadata_lookup(metadata_df: pd.DataFrame) -> pd.DataFrame:
    return metadata_df[
        [
            "dataset",
            "lang_display",
            "utterance_key",
            "mixture_utterance",
            "enrolment_speakers_utterance",
            "total_number_of_speaker",
            "is_primary_speaker",
            "speaker_ratio",
            "mixture_ratio",
            "mixture_duration",
            "transcript_length",
            "enrol_ter",
            "enrol_gt_length",
            "is_official_primary",
        ]
    ].copy()


def _status_record(
    model: str,
    metric_key: str,
    csv_path: Path,
    result_root: str,
    status: str,
    row_count: int = 0,
    matched_row_count: int = 0,
    message: str = "",
) -> dict[str, Any]:
    return {
        "model": model,
        "metric_key": metric_key,
        "metric_label": METRIC_SPECS[metric_key]["label"],
        "result_root": result_root,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.upper()),
        "file_path": str(csv_path),
        "row_count": row_count,
        "matched_row_count": matched_row_count,
        "message": message,
    }


def _load_metric_file(
    model: str,
    model_dir: Path,
    metric_key: str,
    metadata_lookup: pd.DataFrame,
    result_root: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = METRIC_SPECS[metric_key]
    csv_path = model_dir / f"{model}{spec['file_suffix']}"
    if not csv_path.is_file():
        return pd.DataFrame(), _status_record(
            model,
            metric_key,
            csv_path,
            result_root,
            "missing",
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return pd.DataFrame(), _status_record(
            model,
            metric_key,
            csv_path,
            result_root,
            "error",
            message=str(exc),
        )

    if df.empty:
        return pd.DataFrame(), _status_record(
            model,
            metric_key,
            csv_path,
            result_root,
            "empty",
        )

    if spec["source_type"] == "ter":
        required = {"mixture_utterance", "enrolment_speakers_utterance", spec["value_column"]}
        if not required.issubset(df.columns):
            return pd.DataFrame(), _status_record(
                model,
                metric_key,
                csv_path,
                result_root,
                "error",
                row_count=len(df),
                message=f"Missing columns: {sorted(required.difference(df.columns))}",
            )
        work = df.copy()
        work["utterance_key"] = work.apply(
            lambda row: make_utterance_key(
                row["mixture_utterance"], row["enrolment_speakers_utterance"]
            ),
            axis=1,
        )
    else:
        required = {"utterance", spec["value_column"]}
        if not required.issubset(df.columns):
            return pd.DataFrame(), _status_record(
                model,
                metric_key,
                csv_path,
                result_root,
                "error",
                row_count=len(df),
                message=f"Missing columns: {sorted(required.difference(df.columns))}",
            )
        work = df.copy()
        if "status" in work.columns and spec["source_type"] in {"utterance"}:
            work = work[work["status"] == "ok"].copy()
        work["utterance_key"] = work["utterance"].astype(str)

    work["metric_value"] = pd.to_numeric(work[spec["value_column"]], errors="coerce")
    work = work.dropna(subset=["metric_value"]).copy()
    # Metric CSVs often duplicate BASE metadata columns; merging full `work` would suffix
    # them (_x/_y) and break the column list below. Keep only keys + metric for the join.
    merge_left = work[["utterance_key", "metric_value"]].copy()
    merged = merge_left.merge(
        metadata_lookup, on="utterance_key", how="left", validate="many_to_one"
    )
    merged = merged[merged["dataset"].isin(DATASET_ORDER)].copy()

    if merged.empty:
        status = "empty"
    else:
        status = "ok"

    metric_rows = merged[
        [
            "dataset",
            "lang_display",
            "utterance_key",
            "mixture_utterance",
            "enrolment_speakers_utterance",
            "total_number_of_speaker",
            "is_primary_speaker",
            "speaker_ratio",
            "mixture_ratio",
            "mixture_duration",
            "transcript_length",
            "enrol_ter",
            "enrol_gt_length",
            "is_official_primary",
            "metric_value",
        ]
    ].copy()
    metric_rows["model"] = model
    metric_rows["metric_key"] = metric_key
    metric_rows["metric_label"] = spec["label"]
    metric_rows["metric_available"] = True
    metric_rows["source_file"] = str(csv_path)
    metric_rows["result_root"] = result_root

    return metric_rows, _status_record(
        model,
        metric_key,
        csv_path,
        result_root,
        status,
        row_count=len(df),
        matched_row_count=len(metric_rows),
    )


@lru_cache(maxsize=8)
def load_dashboard_state(refresh_token: int = 0) -> DashboardState:
    metadata_df, transcript_length_note, enrol_quality_note, enrol_quality_source_path = (
        load_base_metadata()
    )
    metadata_lookup = _base_metadata_lookup(metadata_df)
    model_source_df = _build_model_source_df()

    metric_frames: list[pd.DataFrame] = []
    availability_records: list[dict[str, Any]] = []
    for row in model_source_df.to_dict(orient="records"):
        model = str(row["model"])
        for result_root, dir_key in (
            (RESULT_ROOT_BASE, "base_dir"),
            (RESULT_ROOT_PRIMARY, "primary_dir"),
        ):
            model_dir = _optional_path(row.get(dir_key))
            if model_dir is None:
                continue
            for metric_key in METRIC_SPECS:
                metric_rows, availability = _load_metric_file(
                    model=model,
                    model_dir=model_dir,
                    metric_key=metric_key,
                    metadata_lookup=metadata_lookup,
                    result_root=result_root,
                )
                availability_records.append(availability)
                if not metric_rows.empty:
                    metric_frames.append(metric_rows)

    metric_long_df = (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else _empty_metric_long_df()
    )
    availability_df = (
        pd.DataFrame(availability_records) if availability_records else _empty_availability_df()
    )
    models = model_source_df["model"].astype(str).tolist() if not model_source_df.empty else []
    return DashboardState(
        metadata_df=metadata_df,
        metric_long_df=metric_long_df,
        availability_df=availability_df,
        model_source_df=model_source_df,
        models=models,
        transcript_length_note=transcript_length_note,
        enrol_quality_note=enrol_quality_note,
        enrol_quality_source_path=enrol_quality_source_path,
    )


def resolve_effective_dashboard_view(
    state: DashboardState,
    preset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_root_lookup = _resolve_effective_result_root_lookup(
        state.model_source_df,
        preset_name,
    )

    if state.metric_long_df.empty:
        effective_metric_long_df = _empty_metric_long_df()
    else:
        work = state.metric_long_df.copy()
        work["effective_result_root"] = work["model"].map(effective_root_lookup)
        effective_metric_long_df = work[
            work["effective_result_root"].notna()
            & work["result_root"].eq(work["effective_result_root"])
        ].drop(columns=["effective_result_root"])

    raw_availability_lookup = {
        (str(row["model"]), str(row["metric_key"]), str(row["result_root"])): row
        for row in state.availability_df.to_dict(orient="records")
    }
    resolved_availability_records: list[dict[str, Any]] = []
    model_rows = state.model_source_df.to_dict(orient="records")
    for row in model_rows:
        model = str(row["model"])
        effective_root = effective_root_lookup.get(model)
        for metric_key in METRIC_SPECS:
            if effective_root is None:
                missing_root = RESULT_ROOT_BASE if preset_name == RESULT_ROOT_BASE else RESULT_ROOT_PRIMARY
                resolved_availability_records.append(
                    _status_record(
                        model=model,
                        metric_key=metric_key,
                        csv_path=_expected_metric_csv_path(model, metric_key, missing_root),
                        result_root=missing_root,
                        status="missing",
                    )
                )
                continue
            record = raw_availability_lookup.get((model, metric_key, effective_root))
            if record is None:
                resolved_availability_records.append(
                    _status_record(
                        model=model,
                        metric_key=metric_key,
                        csv_path=_expected_metric_csv_path(model, metric_key, effective_root),
                        result_root=effective_root,
                        status="missing",
                    )
                )
            else:
                resolved_availability_records.append(record)

    effective_availability_df = (
        pd.DataFrame(resolved_availability_records)
        if resolved_availability_records
        else _empty_availability_df()
    )
    return effective_metric_long_df, effective_availability_df


def build_filter_config(
    preset_name: str,
    selected_speakers: list[int],
    speaker_scope: str,
    speaker_ratio_min: int,
    mixture_ratio_min: int | None,
    mixture_duration_max: int | None,
    transcript_length_min: int,
    enrol_quality_max: float | None,
    enrol_gt_length_filter: str,
) -> FilterConfig:
    return FilterConfig(
        preset_name=preset_name,
        selected_speakers=tuple(sorted(selected_speakers)),
        speaker_scope=speaker_scope,
        speaker_ratio_min=speaker_ratio_min,
        mixture_ratio_min=mixture_ratio_min,
        mixture_duration_max=mixture_duration_max,
        transcript_length_min=transcript_length_min,
        enrol_quality_max=enrol_quality_max,
        enrol_gt_length_filter=enrol_gt_length_filter,
    )


def is_custom_subset(preset_name: str, filter_config: FilterConfig) -> bool:
    defaults = PRESET_DEFAULTS[preset_name]
    current = {
        "selected_speakers": list(filter_config.selected_speakers),
        "speaker_scope": filter_config.speaker_scope,
        "speaker_ratio_min": filter_config.speaker_ratio_min,
        "mixture_ratio_min": filter_config.mixture_ratio_min,
        "mixture_duration_max": filter_config.mixture_duration_max,
        "transcript_length_min": filter_config.transcript_length_min,
        "enrol_quality_max": filter_config.enrol_quality_max,
        "enrol_gt_length_filter": filter_config.enrol_gt_length_filter,
    }
    return any(current[key] != defaults[key] for key in FILTER_KEYS)


def filter_metric_rows(
    metric_long_df: pd.DataFrame,
    selected_models: list[str],
    metric_key: str,
    filter_config: FilterConfig,
) -> pd.DataFrame:
    if metric_long_df.empty or not selected_models:
        return metric_long_df.iloc[0:0].copy()

    if is_sim_ui_metric(metric_key):
        work = metric_long_df[
            metric_long_df["model"].isin(selected_models)
            & metric_long_df["metric_key"].isin(SIM_METRIC_KEYS)
        ].copy()
    else:
        work = metric_long_df[
            metric_long_df["model"].isin(selected_models)
            & metric_long_df["metric_key"].eq(metric_key)
        ].copy()

    if work.empty:
        return work

    if filter_config.preset_name == "PRIMARY":
        work = work[work["is_official_primary"]].copy()
    work = work[work["total_number_of_speaker"].isin(filter_config.selected_speakers)].copy()
    if filter_config.speaker_scope == "primary":
        work = work[work["is_primary_speaker"]].copy()
    work = work[work["speaker_ratio"] >= filter_config.speaker_ratio_min / 100.0].copy()
    work = work[work["transcript_length"] > filter_config.transcript_length_min].copy()
    if filter_config.enrol_quality_max is not None:
        work = work[
            work["enrol_ter"].notna() & (work["enrol_ter"] <= float(filter_config.enrol_quality_max))
        ].copy()

    length_mode = str(filter_config.enrol_gt_length_filter)
    if length_mode != "all":
        lengths = pd.to_numeric(work["enrol_gt_length"], errors="coerce")
        if length_mode == "0-5":
            work = work[lengths.between(0, 5, inclusive="both")].copy()
        elif length_mode == "ge5":
            work = work[lengths >= 5].copy()
        elif length_mode == "ge10":
            work = work[lengths >= 10].copy()
        elif length_mode == "ge15":
            work = work[lengths >= 15].copy()
        elif length_mode == "ge20":
            work = work[lengths >= 20].copy()
        else:
            raise ValueError(f"Unsupported enrol_gt_length_filter: {length_mode}")

    if filter_config.mixture_ratio_min is not None:
        work = work[work["mixture_ratio"] >= filter_config.mixture_ratio_min / 100.0].copy()
    if filter_config.mixture_duration_max is not None:
        work = work[work["mixture_duration"] <= filter_config.mixture_duration_max].copy()

    return work


def _sim_combined_status(model: str, availability_df: pd.DataFrame) -> str:
    statuses: list[str] = []
    for mk in SIM_METRIC_KEYS:
        sub = availability_df[
            (availability_df["model"] == model) & (availability_df["metric_key"] == mk)
        ]
        if sub.empty:
            statuses.append("missing")
        else:
            statuses.append(str(sub.iloc[0]["status"]))
    if "missing" in statuses:
        return "missing"
    if "error" in statuses:
        return "error"
    if all(s == "ok" for s in statuses):
        return "ok"
    return "empty"


def _sim_pair_rows(filtered_metric_rows: pd.DataFrame) -> pd.DataFrame:
    """Inner-join enrol-mixture vs enrol-tse on (model, utterance_key)."""
    empty = pd.DataFrame(columns=_SIM_PAIR_COLUMNS)
    mix = filtered_metric_rows[filtered_metric_rows["metric_key"] == "sim_enrol_mixture"]
    tse = filtered_metric_rows[filtered_metric_rows["metric_key"] == "sim_enrol_tse"]
    if mix.empty or tse.empty:
        return empty.copy()

    m = (
        mix.groupby(["model", "utterance_key"], as_index=False)
        .agg(
            {
                "dataset": "first",
                "lang_display": "first",
                "metric_value": "mean",
            }
        )
        .rename(columns={"metric_value": "v_mix"})
    )
    t = (
        tse.groupby(["model", "utterance_key"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "v_tse"})
    )
    paired = m.merge(t, on=["model", "utterance_key"], how="inner")
    if paired.empty:
        return empty.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        paired["uplift_pct"] = np.where(
            np.abs(paired["v_mix"].astype(float)) > 1e-12,
            (paired["v_tse"].astype(float) - paired["v_mix"].astype(float))
            / paired["v_mix"].astype(float)
            * 100.0,
            np.nan,
        )
    return paired


def _sim_slice_stats(p_slice: pd.DataFrame) -> tuple[int, float, float, float]:
    """Returns count (unique utterance_key), mean mix, mean tse, uplift % from group means."""
    if p_slice.empty:
        return 0, float("nan"), float("nan"), float("nan")
    n = int(p_slice["utterance_key"].nunique())
    mm = float(p_slice["v_mix"].mean())
    mt = float(p_slice["v_tse"].mean())
    if np.isnan(mm) or abs(mm) <= 1e-12:
        return n, mm, mt, float("nan")
    uplift = (mt - mm) / mm * 100.0
    return n, mm, mt, float(uplift)


def format_metric_value(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{float(value):.4f}"


def wrap_model_display_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("_", "\n_")


def format_sim_cell(mix_mean: float, tse_mean: float, uplift_pct: float) -> str:
    if np.isnan(mix_mean) or np.isnan(tse_mean) or np.isnan(uplift_pct):
        return "NA"
    arrow = "↑" if uplift_pct > 0 else "↓" if uplift_pct < 0 else ""
    sign = "+" if uplift_pct > 0 else ""
    return f"{mix_mean:.4f} → {tse_mean:.4f} / {sign}{uplift_pct:.4f}% {arrow}".rstrip()


def _sim_group_summary_records(
    paired_df: pd.DataFrame,
    selected_models: list[str],
    availability_df: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped_specs = [("overall", "Overall", None)] + [
        ("dataset", dataset, ("dataset", dataset)) for dataset in DATASET_ORDER
    ] + [("language", lang, ("lang_display", lang)) for lang in LANGUAGE_ORDER]

    for group_type, group_label, filter_spec in grouped_specs:
        for model in selected_models:
            status = _sim_combined_status(model, availability_df)
            p_model = paired_df[paired_df["model"] == model]
            if filter_spec is not None:
                column_name, expected_value = filter_spec
                p_model = p_model[p_model[column_name] == expected_value]

            if status in {"missing", "error"}:
                count_value = np.nan
                mean_value = np.nan
            else:
                count_value, _mm, _mt, mean_value = _sim_slice_stats(p_model)

            records.append(
                {
                    "group_type": group_type,
                    "group_label": group_label,
                    "model": model,
                    "count": count_value,
                    "mean": mean_value,
                    "status": status,
                }
            )

    return pd.DataFrame.from_records(records)


def build_sim_group_summary_table(
    paired_df: pd.DataFrame,
    selected_models: list[str],
    group_column: str,
    group_order: list[str],
    row_label: str,
) -> pd.DataFrame:
    columns = [row_label, "count", *selected_models]
    if paired_df.empty or not selected_models:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for label in group_order:
        sub = paired_df[paired_df[group_column] == label]
        row: dict[str, Any] = {row_label: label}
        row["count"] = int(sub["utterance_key"].nunique()) if len(sub) else 0
        for model in selected_models:
            msub = sub[sub["model"] == model]
            _n, mm, mt, uplift = _sim_slice_stats(msub)
            row[model] = format_sim_cell(mm, mt, uplift)
        rows.append(row)
    return pd.DataFrame(rows)


def _sim_overall_df(
    paired_df: pd.DataFrame,
    selected_models: list[str],
    availability_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in selected_models:
        status = _sim_combined_status(model, availability_df)
        p_model = paired_df[paired_df["model"] == model]
        if status in {"missing", "error"}:
            rows.append(
                {
                    "model": model,
                    "count": np.nan,
                    "mix_mean": np.nan,
                    "tse_mean": np.nan,
                    "mean": np.nan,
                    "status": status,
                }
            )
        else:
            count_v, mm, mt, uplift = _sim_slice_stats(p_model)
            rows.append(
                {
                    "model": model,
                    "count": count_v,
                    "mix_mean": mm,
                    "tse_mean": mt,
                    "mean": uplift,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def aggregate_sim_metric_rows(
    filtered_metric_rows: pd.DataFrame,
    selected_models: list[str],
    availability_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    paired_df = _sim_pair_rows(filtered_metric_rows)
    summary_long_df = _sim_group_summary_records(
        paired_df=paired_df,
        selected_models=selected_models,
        availability_df=availability_df,
    )

    overall_df = _sim_overall_df(paired_df, selected_models, availability_df)

    dataset_summary_df = build_sim_group_summary_table(
        paired_df,
        selected_models,
        group_column="dataset",
        group_order=DATASET_ORDER,
        row_label="dataset",
    )
    language_summary_df = build_sim_group_summary_table(
        paired_df,
        selected_models,
        group_column="lang_display",
        group_order=LANGUAGE_ORDER,
        row_label="language",
    )

    heatmap_order = ["Overall", *DATASET_ORDER, *LANGUAGE_ORDER]
    heatmap_df = summary_long_df.pivot(index="group_label", columns="model", values="mean").reindex(
        heatmap_order
    )

    return {
        "summary_long_df": summary_long_df,
        "overall_df": overall_df,
        "dataset_summary_df": dataset_summary_df,
        "language_summary_df": language_summary_df,
        "heatmap_df": heatmap_df.reindex(columns=selected_models),
    }


def _availability_lookup(availability_df: pd.DataFrame, metric_key: str) -> dict[tuple[str, str], dict[str, Any]]:
    if availability_df.empty:
        return {}
    metric_availability = availability_df[availability_df["metric_key"] == metric_key].copy()
    return {
        (row["model"], row["metric_key"]): row
        for row in metric_availability.to_dict(orient="records")
    }


def _group_summary_records(
    filtered_metric_rows: pd.DataFrame,
    selected_models: list[str],
    metric_key: str,
    availability_df: pd.DataFrame,
) -> pd.DataFrame:
    availability_lookup = _availability_lookup(availability_df, metric_key)
    records: list[dict[str, Any]] = []

    grouped_specs = [("overall", "Overall", None)] + [
        ("dataset", dataset, ("dataset", dataset)) for dataset in DATASET_ORDER
    ] + [("language", lang, ("lang_display", lang)) for lang in LANGUAGE_ORDER]

    for group_type, group_label, filter_spec in grouped_specs:
        for model in selected_models:
            status_row = availability_lookup.get((model, metric_key))
            status = "missing" if status_row is None else status_row["status"]
            model_rows = filtered_metric_rows[filtered_metric_rows["model"] == model]
            if filter_spec is not None:
                column_name, expected_value = filter_spec
                model_rows = model_rows[model_rows[column_name] == expected_value]

            if status in {"missing", "error"}:
                count_value = np.nan
                mean_value = np.nan
            else:
                count_value = int(len(model_rows))
                mean_value = float(model_rows["metric_value"].mean()) if len(model_rows) else np.nan

            records.append(
                {
                    "group_type": group_type,
                    "group_label": group_label,
                    "model": model,
                    "count": count_value,
                    "mean": mean_value,
                    "status": status,
                }
            )

    return pd.DataFrame.from_records(records)


def build_group_summary_table(
    filtered_metric_rows: pd.DataFrame,
    selected_models: list[str],
    group_column: str,
    group_order: list[str],
    row_label: str,
) -> pd.DataFrame:
    """One row per group: [row_label, count, model_1, ...].

    ``count`` is the number of distinct ``utterance_key`` values in the slice. For TER-style
    rows, ``utterance_key`` is ``mixture_utterance`` + ``enrolment_speakers_utterance``, so
    multiple metadata rows that share the same mixture but differ by enrol/target are **not**
    collapsed. For long-format data (one row per model per key), this equals counting each
    evaluation row once, not ``len(sub)`` which would multiply by the number of models.
    """
    columns = [row_label, "count", *selected_models]
    if filtered_metric_rows.empty or not selected_models:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for label in group_order:
        sub = filtered_metric_rows[filtered_metric_rows[group_column] == label]
        count_val = int(sub["utterance_key"].nunique()) if len(sub) else 0
        row: dict[str, Any] = {row_label: label, "count": count_val}
        for model in selected_models:
            mrows = sub[sub["model"] == model]
            row[model] = float(mrows["metric_value"].mean()) if len(mrows) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metric_rows(
    filtered_metric_rows: pd.DataFrame,
    selected_models: list[str],
    metric_key: str,
    availability_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if is_sim_ui_metric(metric_key):
        return aggregate_sim_metric_rows(
            filtered_metric_rows=filtered_metric_rows,
            selected_models=selected_models,
            availability_df=availability_df,
        )

    summary_long_df = _group_summary_records(
        filtered_metric_rows=filtered_metric_rows,
        selected_models=selected_models,
        metric_key=metric_key,
        availability_df=availability_df,
    )

    overall_df = (
        summary_long_df[summary_long_df["group_type"] == "overall"][
            ["model", "count", "mean", "status"]
        ]
        .set_index("model")
        .reindex(selected_models)
        .reset_index()
    )

    dataset_summary_df = build_group_summary_table(
        filtered_metric_rows,
        selected_models,
        group_column="dataset",
        group_order=DATASET_ORDER,
        row_label="dataset",
    )
    language_summary_df = build_group_summary_table(
        filtered_metric_rows,
        selected_models,
        group_column="lang_display",
        group_order=LANGUAGE_ORDER,
        row_label="language",
    )

    heatmap_order = ["Overall", *DATASET_ORDER, *LANGUAGE_ORDER]
    heatmap_df = summary_long_df.pivot(index="group_label", columns="model", values="mean").reindex(
        heatmap_order
    )

    return {
        "summary_long_df": summary_long_df,
        "overall_df": overall_df,
        "dataset_summary_df": dataset_summary_df,
        "language_summary_df": language_summary_df,
        "heatmap_df": heatmap_df.reindex(columns=selected_models),
    }


def format_group_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format [label, count, *models] for display."""
    if df.empty:
        return df
    display_df = df.copy()
    cols = list(display_df.columns)
    if len(cols) >= 2 and cols[1] == "count":
        display_df["count"] = display_df["count"].map(
            lambda value: "NA" if pd.isna(value) else str(int(value)),
        )
    for col in cols[2:]:
        display_df[col] = display_df[col].map(
            lambda value: "NA" if pd.isna(value) else format_metric_value(float(value)),
        )
    if len(cols) > 2:
        display_df = display_df.rename(columns={col: wrap_model_display_name(col) for col in cols[2:]})
    return display_df


def format_overall_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty:
        return overall_df
    display_df = overall_df.copy()
    display_df["model"] = display_df["model"].map(wrap_model_display_name)
    display_df["count"] = display_df["count"].map(
        lambda value: "NA" if pd.isna(value) else str(int(value))
    )
    display_df["mean"] = display_df["mean"].map(
        lambda value: "NA" if pd.isna(value) else format_metric_value(float(value))
    )
    display_df["status"] = display_df["status"].map(lambda value: STATUS_LABELS.get(value, value.upper()))
    return display_df


def format_sim_overall_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty:
        return overall_df
    display_df = overall_df.copy()
    display_df["model"] = display_df["model"].map(wrap_model_display_name)
    display_df["count"] = display_df["count"].map(
        lambda value: "NA" if pd.isna(value) else str(int(value)),
    )
    sim_column = "SIM(enrol-mixture) → SIM(enrol-tse) / improvement ↑"
    display_df[sim_column] = display_df.apply(
        lambda row: format_sim_cell(row["mix_mean"], row["tse_mean"], row["mean"]),
        axis=1,
    )
    display_df["status"] = display_df["status"].map(
        lambda value: STATUS_LABELS.get(value, str(value).upper()),
    )
    return display_df[["model", "count", sim_column, "status"]]


def format_sim_group_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """SIM group tables already use string cells; normalize count display."""
    if df.empty:
        return df
    display_df = df.copy()
    if len(display_df.columns) >= 2 and display_df.columns[1] == "count":
        display_df["count"] = display_df["count"].map(
            lambda value: "NA" if pd.isna(value) else str(int(value)),
        )
    cols = list(display_df.columns)
    if len(cols) > 2:
        display_df = display_df.rename(columns={col: wrap_model_display_name(col) for col in cols[2:]})
    return display_df
