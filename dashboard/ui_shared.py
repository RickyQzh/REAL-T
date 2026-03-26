from __future__ import annotations

from typing import Any

import streamlit as st

try:
    from dashboard.data import (
        FILTER_KEYS,
        PRESET_DEFAULTS,
        SPEAKER_COUNT_OPTIONS,
        iter_selectable_metric_keys,
        normalize_sidebar_metric_key,
    )
except ModuleNotFoundError:
    from data import (
        FILTER_KEYS,
        PRESET_DEFAULTS,
        SPEAKER_COUNT_OPTIONS,
        iter_selectable_metric_keys,
        normalize_sidebar_metric_key,
    )


AUTO_REFRESH_OPTIONS = {
    "Pause": 0,
    "15s": 15,
    "30s": 30,
    "60s": 60,
    "120s": 120,
}

SPEAKER_SCOPE_OPTIONS = {
    "全部说话人": "all",
    "主说话人": "primary",
}
SPEAKER_SCOPE_LABELS = {value: key for key, value in SPEAKER_SCOPE_OPTIONS.items()}

SPEAKER_RATIO_OPTIONS = [20, 30, 40, 50, 60, 70, 80]
MIXTURE_RATIO_OPTIONS = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]
MIXTURE_DURATION_OPTIONS = [None, 10, 20, 30, 40, 50, 60]
TRANSCRIPT_LENGTH_OPTIONS = [5, 10, 20, 30, 40, 50]
ENROL_QUALITY_OPTIONS = [None, 0.05, 0.10, 0.15, 0.20, 0.30]
ENROL_GT_LENGTH_OPTIONS = ["all", "0-5", "ge5", "ge10", "ge15", "ge20"]
ENROL_GT_LENGTH_LABELS = {
    "all": "不限",
    "0-5": "0-5",
    "ge5": ">= 5",
    "ge10": ">= 10",
    "ge15": ">= 15",
    "ge20": ">= 20",
}

DEFAULT_SELECTED_MODELS = [
    "bsrnn_vox1",
    "tfmap_context_vox_old",
    "lauratse_enrol5",
    "alphaflowtse_noisy_ECAPAMLP_steps1",
]

PAGE_REFS: dict[str, Any] = {}
_NAV_HEADER_CSS = """
<style>
.realt-nav-action-wrap {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding-top: 1.15rem;
}
.realt-nav-action {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    min-width: 11rem;
    padding: 0.9rem 1.4rem;
    border-radius: 999px;
    border: 1px solid rgba(151, 166, 195, 0.45);
    background: rgba(24, 31, 46, 0.88);
    color: #f5f7fb !important;
    font-size: 1.12rem;
    font-weight: 700;
    line-height: 1;
    text-decoration: none !important;
    transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
}
.realt-nav-action:hover {
    transform: translateY(-1px);
    background: rgba(34, 43, 63, 0.98);
    border-color: rgba(170, 186, 220, 0.72);
}
</style>
"""


def apply_preset_to_state(preset_name: str) -> None:
    defaults = PRESET_DEFAULTS[preset_name]
    for key in FILTER_KEYS:
        value = defaults[key]
        st.session_state[key] = value.copy() if isinstance(value, list) else value
    st.session_state["speaker_scope_label"] = SPEAKER_SCOPE_LABELS[defaults["speaker_scope"]]


def handle_preset_change() -> None:
    apply_preset_to_state(st.session_state["preset_name"])


def ensure_session_defaults() -> None:
    if "preset_name" not in st.session_state:
        st.session_state["preset_name"] = "PRIMARY"
        apply_preset_to_state("PRIMARY")
    else:
        for key in FILTER_KEYS:
            if key not in st.session_state:
                default_value = PRESET_DEFAULTS[st.session_state["preset_name"]][key]
                st.session_state[key] = (
                    default_value.copy() if isinstance(default_value, list) else default_value
                )

    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = next(iter(iter_selectable_metric_keys()))
    else:
        st.session_state["selected_metric"] = normalize_sidebar_metric_key(
            st.session_state["selected_metric"]
        )
        valid = set(iter_selectable_metric_keys())
        if st.session_state["selected_metric"] not in valid:
            st.session_state["selected_metric"] = next(iter(iter_selectable_metric_keys()))
    if "auto_refresh_label" not in st.session_state:
        st.session_state["auto_refresh_label"] = "Pause"
    if "bar_group_label" not in st.session_state:
        st.session_state["bar_group_label"] = "Overall"
    if "speaker_scope_label" not in st.session_state:
        st.session_state["speaker_scope_label"] = SPEAKER_SCOPE_LABELS[st.session_state["speaker_scope"]]


def format_optional_threshold(value: int | None, prefix: str) -> str:
    if value is None:
        return "不限"
    return f"{prefix}{value}"


def default_model_selection(model_options: list[str]) -> list[str]:
    preferred = [model for model in DEFAULT_SELECTED_MODELS if model in model_options]
    return preferred if preferred else list(model_options)


def render_page_header(title: str, action_label: str, action_href: str, *, reverse: bool = False) -> None:
    st.markdown(_NAV_HEADER_CSS, unsafe_allow_html=True)
    title_col, action_col = st.columns([5.3, 1.35])
    with title_col:
        st.title(title)
    arrow = "&larr;" if reverse else "&rarr;"
    if reverse:
        action_html = f"""
        <div class="realt-nav-action-wrap">
          <a class="realt-nav-action" href="{action_href}" target="_self">
            <span aria-hidden="true">{arrow}</span><span>{action_label}</span>
          </a>
        </div>
        """
    else:
        action_html = f"""
        <div class="realt-nav-action-wrap">
          <a class="realt-nav-action" href="{action_href}" target="_self">
            <span>{action_label}</span><span aria-hidden="true">{arrow}</span>
          </a>
        </div>
        """
    with action_col:
        st.markdown(action_html, unsafe_allow_html=True)


def build_filter_summary_items(
    filter_config: Any,
    preset_name: str,
    *,
    custom_subset: bool,
) -> list[str]:
    subset_label = (
        f"Custom subset (based on {preset_name})" if custom_subset else preset_name
    )
    return [
        f"Subset: {subset_label}",
        f"Speaker ratio >= {filter_config.speaker_ratio_min}%",
        f"Transcript length > {filter_config.transcript_length_min}",
        (
            f"Enrol TER <= {filter_config.enrol_quality_max:.4f}"
            if filter_config.enrol_quality_max is not None
            else "Enrol TER: 不限"
        ),
        f"Enrol GT length: {ENROL_GT_LENGTH_LABELS.get(filter_config.enrol_gt_length_filter, filter_config.enrol_gt_length_filter)}",
        f"Speaker scope: {SPEAKER_SCOPE_LABELS[filter_config.speaker_scope]}",
        (
            f"Mixture ratio: {format_optional_threshold(filter_config.mixture_ratio_min, '>= ')}%"
            if filter_config.mixture_ratio_min is not None
            else "Mixture ratio: 不限"
        ),
        (
            f"Mixture duration: {format_optional_threshold(filter_config.mixture_duration_max, '<= ')}s"
            if filter_config.mixture_duration_max is not None
            else "Mixture duration: 不限"
        ),
    ]
