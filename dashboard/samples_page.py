from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, APP_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from dashboard.data import (
        DATASET_ORDER,
        SAMPLE_METRIC_COLUMNS,
        build_filter_config,
        filter_sample_rows,
        is_custom_subset,
        load_dashboard_catalog,
        load_effective_sample_detail_for_model,
    )
    from dashboard.ui_shared import (
        ENROL_GT_LENGTH_LABELS,
        ENROL_GT_LENGTH_OPTIONS,
        ENROL_QUALITY_OPTIONS,
        MIXTURE_DURATION_OPTIONS,
        MIXTURE_RATIO_OPTIONS,
        SPEAKER_COUNT_OPTIONS,
        SPEAKER_RATIO_OPTIONS,
        SPEAKER_SCOPE_OPTIONS,
        TRANSCRIPT_LENGTH_OPTIONS,
        build_filter_summary_items,
        default_model_selection,
        ensure_session_defaults,
        handle_preset_change,
        render_page_header,
    )
except ModuleNotFoundError:
    from data import (
        DATASET_ORDER,
        SAMPLE_METRIC_COLUMNS,
        build_filter_config,
        filter_sample_rows,
        is_custom_subset,
        load_dashboard_catalog,
        load_effective_sample_detail_for_model,
    )
    from ui_shared import (
        ENROL_GT_LENGTH_LABELS,
        ENROL_GT_LENGTH_OPTIONS,
        ENROL_QUALITY_OPTIONS,
        MIXTURE_DURATION_OPTIONS,
        MIXTURE_RATIO_OPTIONS,
        SPEAKER_COUNT_OPTIONS,
        SPEAKER_RATIO_OPTIONS,
        SPEAKER_SCOPE_OPTIONS,
        TRANSCRIPT_LENGTH_OPTIONS,
        build_filter_summary_items,
        default_model_selection,
        ensure_session_defaults,
        handle_preset_change,
        render_page_header,
    )

PAGE_SIZE = 20
SORT_METRIC_OPTIONS = [
    "utterance_key",
    "ter_whisper",
    "ter_fireredasr2",
    "sim_enrol_mixture",
    "sim_enrol_tse",
    "sim_uplift_pct",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "dnsmos_p808",
    "ratio_precision",
    "ratio_recall",
    "ratio_f1",
    "enrol_ter",
    "enrol_gt_length",
    "speaker_ratio",
    "mixture_ratio",
    "mixture_duration",
    "transcript_length",
]
SORT_METRIC_LABELS = {
    "utterance_key": "Sample",
    "ter_whisper": "TER / whisper",
    "ter_fireredasr2": "TER / fireredasr2",
    "sim_enrol_mixture": "SIM / enrol-mixture",
    "sim_enrol_tse": "SIM / enrol-tse",
    "sim_uplift_pct": "SIM uplift %",
    "dnsmos_sig": "DNSMOS / SIG",
    "dnsmos_bak": "DNSMOS / BAK",
    "dnsmos_ovrl": "DNSMOS / OVRL",
    "dnsmos_p808": "DNSMOS / P808",
    "ratio_precision": "RATIO / precision",
    "ratio_recall": "RATIO / recall",
    "ratio_f1": "RATIO / f1",
    "enrol_ter": "Enrol TER",
    "enrol_gt_length": "Enrol GT length",
    "speaker_ratio": "speaker_ratio",
    "mixture_ratio": "mixture_ratio",
    "mixture_duration": "mixture_duration",
    "transcript_length": "transcript_length",
}
SORT_ORDER_OPTIONS = {
    "Descending": False,
    "Ascending": True,
}


def _inject_samples_page_styles() -> None:
    st.markdown(
        """
        <style>
        .realt-sample-header {
            padding: 1.05rem 1.15rem 0.7rem 1.15rem;
            margin: 1rem 0 0.55rem 0;
            border: 1px solid rgba(34, 197, 94, 0.20);
            border-radius: 20px;
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 34%),
                linear-gradient(180deg, rgba(30, 41, 59, 0.84), rgba(15, 23, 42, 0.98));
            box-shadow: 0 18px 44px rgba(2, 6, 23, 0.26);
        }
        .realt-sample-title {
            margin: 0;
            font-size: 1.16rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            color: #f8fafc;
            word-break: break-word;
        }
        .realt-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }
        .realt-meta-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #cbd5e1;
            font-size: 0.79rem;
            line-height: 1.25;
        }
        .realt-audio-label {
            margin: 0.2rem 0 0.4rem 0;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #e2e8f0;
            text-align: center;
        }
        .realt-metric-card {
            height: 100%;
            min-height: 132px;
            padding: 0.78rem 0.88rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.08);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.28), rgba(15, 23, 42, 0.56));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.015);
        }
        .realt-metric-title {
            margin: 0 0 0.55rem 0;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #94a3b8;
        }
        .realt-metric-line {
            margin: 0.18rem 0;
            font-size: 0.88rem;
            line-height: 1.45;
            color: #e2e8f0;
        }
        .realt-metric-line-compact {
            margin: 0.12rem 0;
            font-size: 0.79rem;
            line-height: 1.28;
            color: #e2e8f0;
        }
        .realt-metric-key {
            color: #64748b;
            margin-right: 0.28rem;
        }
        .realt-text-card {
            margin-top: 0.7rem;
            padding: 0.85rem 0.95rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background: rgba(15, 23, 42, 0.58);
        }
        .realt-text-label {
            margin: 0 0 0.4rem 0;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #a5f3fc;
        }
        .realt-transcript-card {
            margin-top: 0.7rem;
            padding: 0.78rem 0.88rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.08);
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.28), rgba(15, 23, 42, 0.56));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.015);
        }
        .realt-transcript-title {
            margin: 0 0 0.75rem 0;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #94a3b8;
        }
        .realt-transcript-row {
            display: grid;
            grid-template-columns: max-content 1fr;
            gap: 0.75rem;
            align-items: start;
            padding: 0.28rem 0;
        }
        .realt-transcript-row + .realt-transcript-row {
            border-top: 1px solid rgba(148, 163, 184, 0.08);
            margin-top: 0.3rem;
            padding-top: 0.6rem;
        }
        .realt-transcript-label {
            font-size: 0.68rem;
            line-height: 1.55;
            color: #94a3b8;
            text-align: left;
            white-space: normal;
        }
        .realt-transcript-body {
            font-size: 0.56rem;
            line-height: 1.7;
            color: #e2e8f0;
            word-break: break-word;
        }
        .realt-diff-miss {
            background: rgba(248, 113, 113, 0.12);
            text-decoration: line-through;
            text-decoration-color: rgba(248, 113, 113, 0.85);
            border-radius: 4px;
            padding: 0 0.1rem;
        }
        .realt-diff-extra {
            background: rgba(251, 191, 36, 0.12);
            text-decoration: underline;
            text-decoration-color: rgba(251, 191, 36, 0.85);
            text-underline-offset: 0.14rem;
            border-radius: 4px;
            padding: 0 0.1rem;
        }
        .realt-diff-replace-ref {
            background: rgba(244, 114, 182, 0.12);
            text-decoration: line-through;
            text-decoration-color: rgba(244, 114, 182, 0.88);
            border-radius: 4px;
            padding: 0 0.1rem;
        }
        .realt-diff-replace-hyp {
            background: rgba(96, 165, 250, 0.14);
            text-decoration: underline;
            text-decoration-color: rgba(96, 165, 250, 0.88);
            text-underline-offset: 0.14rem;
            border-radius: 4px;
            padding: 0 0.1rem;
        }
        .realt-transcript-legend {
            margin-top: 0.55rem;
            font-size: 0.62rem;
            color: #64748b;
            line-height: 1.5;
        }
        .realt-transcript-legend span {
            display: inline-block;
            margin-right: 0.55rem;
        }
        @media (max-width: 980px) {
            .realt-transcript-row {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }
            .realt-transcript-label {
                text-align: left;
            }
        }
        .realt-sample-divider {
            height: 1px;
            margin: 1.15rem 0 1.35rem 0;
            background: linear-gradient(90deg, rgba(34,197,94,0.0), rgba(34,197,94,0.35), rgba(34,197,94,0.0));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _format_int_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(value))


def _format_percent_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}%"


def _format_text_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _tokenize_transcript(text: str) -> list[str]:
    if _contains_cjk(text):
        tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*|[^\w\s]|\s+", text)
    else:
        tokens = re.findall(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*|[^\w\s]|\s+", text)
    return tokens if tokens else [text]


def _append_diff_segment(parts: list[str], token: str, css_class: str | None) -> None:
    escaped = html.escape(token)
    if token.isspace():
        parts.append(token.replace(" ", "&nbsp;").replace("\n", "<br>"))
        return
    if css_class:
        parts.append(f'<span class="{css_class}">{escaped}</span>')
    else:
        parts.append(escaped)


def _highlight_transcript_diff(reference: str, hypothesis: str) -> tuple[str, str]:
    ref_tokens = _tokenize_transcript(reference)
    hyp_tokens = _tokenize_transcript(hypothesis)
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    ref_parts: list[str] = []
    hyp_parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for token in ref_tokens[i1:i2]:
                _append_diff_segment(ref_parts, token, None)
            for token in hyp_tokens[j1:j2]:
                _append_diff_segment(hyp_parts, token, None)
        elif tag == "delete":
            for token in ref_tokens[i1:i2]:
                _append_diff_segment(ref_parts, token, "realt-diff-miss")
        elif tag == "insert":
            for token in hyp_tokens[j1:j2]:
                _append_diff_segment(hyp_parts, token, "realt-diff-extra")
        elif tag == "replace":
            for token in ref_tokens[i1:i2]:
                _append_diff_segment(ref_parts, token, "realt-diff-replace-ref")
            for token in hyp_tokens[j1:j2]:
                _append_diff_segment(hyp_parts, token, "realt-diff-replace-hyp")
    return "".join(ref_parts), "".join(hyp_parts)


def _ensure_sample_session_defaults(model_options: list[str], available_datasets: list[str]) -> None:
    preferred_models = st.session_state.get("selected_models") or default_model_selection(model_options)
    default_model = next((model for model in preferred_models if model in model_options), None)
    if default_model is None and model_options:
        default_model = model_options[0]

    current_model = st.session_state.get("samples_selected_model")
    if current_model not in model_options:
        st.session_state["samples_selected_model"] = default_model

    default_dataset = available_datasets[0] if available_datasets else DATASET_ORDER[0]
    current_dataset = st.session_state.get("samples_selected_dataset")
    if current_dataset not in DATASET_ORDER or current_dataset not in available_datasets:
        st.session_state["samples_selected_dataset"] = default_dataset

    if "samples_page_number" not in st.session_state:
        st.session_state["samples_page_number"] = 1
    default_sort_metric = st.session_state.get("selected_metric", "ter_whisper")
    if default_sort_metric == "sim":
        default_sort_metric = "sim_uplift_pct"
    elif default_sort_metric not in SORT_METRIC_OPTIONS:
        default_sort_metric = "utterance_key"
    if st.session_state.get("samples_sort_metric") not in SORT_METRIC_OPTIONS:
        st.session_state["samples_sort_metric"] = default_sort_metric
    if "samples_sort_order_label" not in st.session_state:
        if st.session_state["samples_sort_metric"] in {"ter_whisper", "ter_fireredasr2", "utterance_key"}:
            st.session_state["samples_sort_order_label"] = "Ascending"
        else:
            st.session_state["samples_sort_order_label"] = "Descending"


def _sort_sample_rows(sample_rows: pd.DataFrame, sort_metric: str, ascending: bool) -> pd.DataFrame:
    if sample_rows.empty:
        return sample_rows

    work = sample_rows.copy()
    if sort_metric != "utterance_key":
        work["_sort_value"] = pd.to_numeric(work[sort_metric], errors="coerce")
        work = work.sort_values(
            by=["_sort_value", "utterance_key"],
            ascending=[ascending, True],
            na_position="last",
        ).drop(columns=["_sort_value"])
        return work.reset_index(drop=True)
    return work.sort_values("utterance_key", ascending=ascending).reset_index(drop=True)


def _render_audio_block(label: str, path_value: object) -> None:
    st.markdown(f'<div class="realt-audio-label">{html.escape(label)}</div>', unsafe_allow_html=True)
    if path_value is None or pd.isna(path_value) or not str(path_value).strip():
        st.caption("audio missing")
        return

    path = Path(str(path_value))
    if not path.is_file():
        st.caption("audio missing")
        return

    try:
        st.audio(str(path), format="audio/wav")
    except Exception as exc:  # pragma: no cover - depends on Streamlit runtime
        st.caption(f"audio unavailable: {exc}")


def _metric_card_html(title: str, items: list[tuple[str, str]], compact: bool = False) -> str:
    line_class = "realt-metric-line-compact" if compact else "realt-metric-line"
    lines = "".join(
        (
            f'<div class="{line_class}"><span class="realt-metric-key">{html.escape(key)}</span>'
            f"{html.escape(value)}</div>"
        )
        for key, value in items
    )
    return (
        f'<div class="realt-metric-card"><div class="realt-metric-title">{html.escape(title)}</div>'
        f"{lines}</div>"
    )


def _render_metric_summary(row: pd.Series) -> None:
    metric_columns = st.columns(5)
    metric_cards = [
        (
            "TER",
            [
                ("ASR1", _format_metric_cell(row.get("ter_whisper"))),
                ("ASR2", _format_metric_cell(row.get("ter_fireredasr2"))),
            ],
        ),
        (
            "SIM",
            [
                ("Enrol + Mix", _format_metric_cell(row.get("sim_enrol_mixture"))),
                ("Enrol + TSE", _format_metric_cell(row.get("sim_enrol_tse"))),
                ("Uplift", _format_percent_cell(row.get("sim_uplift_pct"))),
            ],
        ),
        (
            "DNSMOS",
            [
                ("SIG", _format_metric_cell(row.get("dnsmos_sig"))),
                ("BAK", _format_metric_cell(row.get("dnsmos_bak"))),
                ("OVRL", _format_metric_cell(row.get("dnsmos_ovrl"))),
                ("P808", _format_metric_cell(row.get("dnsmos_p808"))),
            ],
        ),
        (
            "RATIO",
            [
                ("Precision", _format_metric_cell(row.get("ratio_precision"))),
                ("Recall", _format_metric_cell(row.get("ratio_recall"))),
                ("F1", _format_metric_cell(row.get("ratio_f1"))),
            ],
        ),
        (
            "ENROLL",
            [
                ("TER", _format_metric_cell(row.get("enrol_ter"))),
                ("GT Length", _format_int_cell(row.get("enrol_gt_length"))),
            ],
        ),
    ]
    for column, (title, items) in zip(metric_columns, metric_cards):
        with column:
            st.markdown(
                _metric_card_html(title, items, compact=(title == "DNSMOS")),
                unsafe_allow_html=True,
            )


def _render_transcript_card(ground_truth: object, tse_transcript: object) -> None:
    ground_truth_text = _format_text_cell(ground_truth)
    tse_transcript_text = _format_text_cell(tse_transcript)
    if ground_truth_text == "NA" or tse_transcript_text == "NA":
        gt_html = html.escape(ground_truth_text)
        tse_html = html.escape(tse_transcript_text)
    else:
        gt_html, tse_html = _highlight_transcript_diff(ground_truth_text, tse_transcript_text)
    st.markdown(
        (
            '<div class="realt-transcript-card">'
            '<div class="realt-transcript-title">TRANSCRIPT</div>'
            '<div class="realt-transcript-row">'
            '<div class="realt-transcript-label">Ground truth transcript :</div>'
            f'<div class="realt-transcript-body">{gt_html}</div>'
            "</div>"
            '<div class="realt-transcript-row">'
            '<div class="realt-transcript-label">TSE transcript by ASR1 :</div>'
            f'<div class="realt-transcript-body">{tse_html}</div>'
            "</div>"
            '<div class="realt-transcript-legend">'
            '<span><span class="realt-diff-miss">missing</span> GT words missed by ASR</span>'
            '<span><span class="realt-diff-extra">extra</span> ASR insertions</span>'
            '<span><span class="realt-diff-replace-hyp">replace</span> substituted words</span>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_sample_card(row: pd.Series) -> None:
    meta_items = [
        f"dataset: {row['dataset']}",
        f"lang: {row['lang_display']}",
        (
            f"speakers: {int(row['total_number_of_speaker'])}"
            if pd.notna(row["total_number_of_speaker"])
            else "speakers: NA"
        ),
        f"speaker ratio: {_format_metric_cell(row['speaker_ratio'])}",
        f"mixture ratio: {_format_metric_cell(row['mixture_ratio'])}",
        f"duration: {_format_metric_cell(row['mixture_duration'])}s",
        (
            f"transcript length: {int(row['transcript_length'])}"
            if pd.notna(row["transcript_length"])
            else "transcript length: NA"
        ),
    ]
    meta_html = "".join(f'<span class="realt-meta-pill">{html.escape(item)}</span>' for item in meta_items)
    st.markdown(
        f'<div class="realt-sample-header"><div class="realt-sample-title">{html.escape(str(row["utterance_key"]))}</div>'
        f'<div class="realt-meta-row">{meta_html}</div></div>',
        unsafe_allow_html=True,
    )

    audio_cols = st.columns(3)
    with audio_cols[0]:
        _render_audio_block("Mixture", row.get("mixture_audio_path"))
    with audio_cols[1]:
        _render_audio_block("Enrol", row.get("enrol_audio_path"))
    with audio_cols[2]:
        _render_audio_block("TSE", row.get("tse_audio_path"))

    _render_metric_summary(row)
    _render_transcript_card(row.get("ground_truth_transcript"), row.get("tse_transcript_asr1"))
    st.markdown('<div class="realt-sample-divider"></div>', unsafe_allow_html=True)


def _inject_single_audio_playback_guard() -> None:
    components.html(
        """
        <script>
        (function() {
          const parentWindow = window.parent;
          if (!parentWindow) {
            return;
          }

          const bindAudioGuard = () => {
            const audios = parentWindow.document.querySelectorAll("audio");
            audios.forEach((audio) => {
              if (audio.dataset.realtSingleAudioBound === "1") {
                return;
              }
              audio.dataset.realtSingleAudioBound = "1";
              audio.addEventListener("play", () => {
                parentWindow.document.querySelectorAll("audio").forEach((other) => {
                  if (other !== audio && !other.paused) {
                    other.pause();
                  }
                });
              });
            });
          };

          bindAudioGuard();
          if (!parentWindow.__realtSingleAudioGuardInterval) {
            parentWindow.__realtSingleAudioGuardInterval = parentWindow.setInterval(
              bindAudioGuard,
              1000
            );
          }
        })();
        </script>
        """,
        height=0,
    )


@st.cache_resource(show_spinner=False)
def get_samples_catalog_cached(refresh_token: int) -> object:
    return load_dashboard_catalog(refresh_token)


@st.cache_resource(show_spinner=False)
def get_sample_rows_cached(model: str, preset_name: str, refresh_token: int) -> pd.DataFrame:
    return load_effective_sample_detail_for_model(model, preset_name, refresh_token)


def clear_samples_page_runtime_caches() -> None:
    get_samples_catalog_cached.clear()
    get_sample_rows_cached.clear()


def render_samples_page() -> None:
    ensure_session_defaults()
    refresh_token = 0
    state = get_samples_catalog_cached(refresh_token)
    _inject_samples_page_styles()

    if not state.models:
        st.error("No model directories were found under `output/BASE` or `output/PRIMARY`.")
        st.stop()

    _ensure_sample_session_defaults(state.models, DATASET_ORDER)

    with st.sidebar:
        st.header("Filters")
        st.radio(
            "预设",
            options=["BASE", "PRIMARY"],
            key="preset_name",
            on_change=handle_preset_change,
            horizontal=True,
        )
        st.selectbox(
            "模型",
            options=state.models,
            key="samples_selected_model",
        )
        st.multiselect(
            "total_num_of_speaker",
            options=SPEAKER_COUNT_OPTIONS,
            default=st.session_state["selected_speakers"],
            key="selected_speakers",
        )
        st.radio(
            "目标说话人范围",
            options=list(SPEAKER_SCOPE_OPTIONS.keys()),
            index=list(SPEAKER_SCOPE_OPTIONS.values()).index(st.session_state["speaker_scope"]),
            key="speaker_scope_label",
            horizontal=True,
        )
        st.session_state["speaker_scope"] = SPEAKER_SCOPE_OPTIONS[st.session_state["speaker_scope_label"]]
        st.selectbox(
            "speaker_ratio",
            options=SPEAKER_RATIO_OPTIONS,
            format_func=lambda value: f">= {value}%",
            key="speaker_ratio_min",
        )
        st.selectbox(
            "mixture_ratio",
            options=MIXTURE_RATIO_OPTIONS,
            format_func=lambda value: "不限" if value is None else f">= {value}%",
            key="mixture_ratio_min",
        )
        st.selectbox(
            "mixture_duration",
            options=MIXTURE_DURATION_OPTIONS,
            format_func=lambda value: "不限" if value is None else f"<= {value}s",
            key="mixture_duration_max",
        )
        st.selectbox(
            "ground_truth_transcript 长度",
            options=TRANSCRIPT_LENGTH_OPTIONS,
            format_func=lambda value: f"> {value}",
            key="transcript_length_min",
        )
        st.selectbox(
            "enrol quality (TER)",
            options=ENROL_QUALITY_OPTIONS,
            format_func=lambda value: "不限" if value is None else f"<= {float(value):.4f}",
            key="enrol_quality_max",
        )
        st.selectbox(
            "enrol length (GT词数)",
            options=ENROL_GT_LENGTH_OPTIONS,
            format_func=lambda value: ENROL_GT_LENGTH_LABELS.get(value, str(value)),
            key="enrol_gt_length_filter",
        )

    filter_config = build_filter_config(
        preset_name=st.session_state["preset_name"],
        selected_speakers=st.session_state["selected_speakers"],
        speaker_scope=st.session_state["speaker_scope"],
        speaker_ratio_min=st.session_state["speaker_ratio_min"],
        mixture_ratio_min=st.session_state["mixture_ratio_min"],
        mixture_duration_max=st.session_state["mixture_duration_max"],
        transcript_length_min=st.session_state["transcript_length_min"],
        enrol_quality_max=st.session_state["enrol_quality_max"],
        enrol_gt_length_filter=st.session_state["enrol_gt_length_filter"],
    )

    sample_detail_df = get_sample_rows_cached(
        st.session_state["samples_selected_model"],
        filter_config.preset_name,
        refresh_token,
    )
    model_filtered_rows = filter_sample_rows(
        sample_detail_df,
        st.session_state["samples_selected_model"],
        filter_config,
    )
    available_datasets = [
        dataset
        for dataset in DATASET_ORDER
        if not model_filtered_rows[model_filtered_rows["dataset"].eq(dataset)].empty
    ]
    _ensure_sample_session_defaults(state.models, available_datasets or DATASET_ORDER)

    with st.sidebar:
        st.selectbox(
            "dataset",
            options=DATASET_ORDER,
            key="samples_selected_dataset",
        )

    render_page_header("REAL-T Samples", "Dashboard", "./", reverse=True)
    _inject_single_audio_playback_guard()

    summary_items = build_filter_summary_items(
        filter_config,
        st.session_state["preset_name"],
        custom_subset=is_custom_subset(st.session_state["preset_name"], filter_config),
    )
    summary_items.extend(
        [
            f"Model: {st.session_state['samples_selected_model']}",
            f"Dataset: {st.session_state['samples_selected_dataset']}",
            f"Sort metric: {SORT_METRIC_LABELS[st.session_state['samples_sort_metric']]}",
            f"Sort order: {st.session_state['samples_sort_order_label']}",
        ]
    )
    st.info(" | ".join(summary_items))

    sample_rows = model_filtered_rows[
        model_filtered_rows["dataset"].eq(st.session_state["samples_selected_dataset"])
    ].copy()
    sample_rows = _sort_sample_rows(
        sample_rows,
        st.session_state["samples_sort_metric"],
        SORT_ORDER_OPTIONS[st.session_state["samples_sort_order_label"]],
    )

    if sample_rows.empty:
        st.warning("No samples matched the current filters.")
        st.stop()

    controls_left, controls_mid, controls_right = st.columns([1.2, 1.0, 1.0])
    with controls_left:
        st.selectbox(
            "Sort metric",
            options=SORT_METRIC_OPTIONS,
            key="samples_sort_metric",
            format_func=lambda value: SORT_METRIC_LABELS.get(value, value),
        )
    with controls_mid:
        st.selectbox(
            "Sort order",
            options=list(SORT_ORDER_OPTIONS.keys()),
            key="samples_sort_order_label",
        )

    total_pages = max(1, math.ceil(len(sample_rows) / PAGE_SIZE))
    if st.session_state["samples_page_number"] > total_pages:
        st.session_state["samples_page_number"] = total_pages

    page_options = list(range(1, total_pages + 1))
    with controls_right:
        st.selectbox(
            "Page",
            options=page_options,
            key="samples_page_number",
            format_func=lambda value: f"{value} ({(value - 1) * PAGE_SIZE + 1}-{min(value * PAGE_SIZE, len(sample_rows))})",
        )

    page_number = int(st.session_state["samples_page_number"])
    start_idx = (page_number - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    visible_rows = sample_rows.iloc[start_idx:end_idx].reset_index(drop=True)

    st.caption(
        f"Showing {start_idx + 1}-{min(end_idx, len(sample_rows))} of {len(sample_rows)} samples"
    )
    for _, row in visible_rows.iterrows():
        _render_sample_card(row)


if __name__ == "__main__":
    render_samples_page()
