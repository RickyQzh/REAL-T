from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
        FILTER_KEYS,
        LANGUAGE_ORDER,
        METRIC_SPECS,
        PRESET_DEFAULTS,
        SIM_METRIC_KEYS,
        SPEAKER_COUNT_OPTIONS,
        aggregate_metric_rows,
        build_filter_config,
        filter_metric_rows,
        format_group_summary_table,
        format_overall_table,
        format_sim_group_summary_table,
        format_sim_overall_table,
        is_custom_subset,
        is_sim_ui_metric,
        iter_selectable_metric_keys,
        load_dashboard_state,
        normalize_sidebar_metric_key,
        selectable_metric_label,
    )
except ModuleNotFoundError:
    from data import (
        DATASET_ORDER,
        FILTER_KEYS,
        LANGUAGE_ORDER,
        METRIC_SPECS,
        PRESET_DEFAULTS,
        SIM_METRIC_KEYS,
        SPEAKER_COUNT_OPTIONS,
        aggregate_metric_rows,
        build_filter_config,
        filter_metric_rows,
        format_group_summary_table,
        format_overall_table,
        format_sim_group_summary_table,
        format_sim_overall_table,
        is_custom_subset,
        is_sim_ui_metric,
        iter_selectable_metric_keys,
        load_dashboard_state,
        normalize_sidebar_metric_key,
        selectable_metric_label,
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


st.set_page_config(
    page_title="REAL-T BASE Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


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
        st.session_state["preset_name"] = "BASE"
        apply_preset_to_state("BASE")
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
    # 默认 Pause：否则 sidebar 里「自动刷新」为 30s 时，每 30s 会整页 reload（schedule_auto_refresh）。
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


def schedule_auto_refresh(interval_seconds: int) -> None:
    if interval_seconds <= 0:
        return
    milliseconds = interval_seconds * 1000
    components.html(
        f"""
        <script>
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, {milliseconds});
        </script>
        """,
        height=0,
        width=0,
    )


def build_heatmap(
    mean_df: pd.DataFrame,
    *,
    lower_is_better: bool = False,
    sim_uplift: bool = False,
) -> go.Figure:
    matrix = mean_df.copy()
    if sim_uplift:
        labels = matrix.map(
            lambda value: "" if pd.isna(value) else f"{float(value):+.2f}%",
        )
        hover_z = "%{z:+.2f}%"
        cbar_title = "Δ% (↑ better)"
        cbar_fmt: dict[str, str] = {"tickformat": "+.2f", "ticksuffix": "%"}
        reverse = False
    else:
        labels = matrix.map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
        hover_z = "%{z:.2f}"
        cbar_title = (
            "Mean (lower is better)" if lower_is_better else "Mean (higher is better)"
        )
        cbar_fmt = {"tickformat": ".2f"}
        reverse = lower_is_better

    # Blues: high z → dark blue. TER: reversescale so small z is dark blue. SIM Δ%: higher
    # uplift → dark blue (default direction).
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            text=labels.values,
            texttemplate="%{text}",
            colorscale="Blues",
            reversescale=reverse,
            hoverongaps=False,
            hovertemplate=f"%{{y}}<br>%{{x}}<br>{hover_z}<extra></extra>",
            colorbar={"title": cbar_title, **cbar_fmt},
        )
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=420,
    )
    return figure


def build_group_bar_chart(
    summary_long_df: pd.DataFrame,
    group_label: str,
    *,
    sim_uplift: bool = False,
) -> go.Figure:
    group_df = summary_long_df[summary_long_df["group_label"] == group_label].copy()
    chart_df = group_df.dropna(subset=["mean"]).copy()
    y_title = "Δ% (tse vs enrol-mixture)" if sim_uplift else "Mean"

    figure = go.Figure()
    if chart_df.empty:
        figure.update_layout(
            title=f"{group_label}: no available values",
            xaxis_title="Model",
            yaxis_title=y_title,
            height=360,
        )
        return figure

    if sim_uplift:
        text_vals = chart_df["mean"].map(lambda value: f"{value:+.2f}%")
        hover_y = "%{y:+.2f}%"
    else:
        text_vals = chart_df["mean"].map(lambda value: f"{value:.2f}")
        hover_y = "%{y:.2f}"

    figure.add_bar(
        x=chart_df["model"],
        y=chart_df["mean"],
        text=text_vals,
        textposition="outside",
        marker_color="#2962ff",
        hovertemplate=f"%{{x}}<br>{hover_y}<extra></extra>",
    )
    yaxis_cfg = (
        {"tickformat": "+.2f", "ticksuffix": "%"} if sim_uplift else {"tickformat": ".2f"}
    )
    figure.update_layout(
        title=group_label,
        xaxis_title="Model",
        yaxis_title=y_title,
        yaxis=yaxis_cfg,
        height=360,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def render_sidebar(model_options: list[str]) -> tuple[list[str], str, int]:
    stored_models = st.session_state.get("selected_models")
    if stored_models is None:
        current_models = list(model_options)
    else:
        current_models = [m for m in stored_models if m in model_options]
        if not current_models:
            current_models = list(model_options)

    with st.sidebar:
        st.header("Filters")
        st.radio(
            "预设",
            options=["BASE", "PRIMARY"],
            key="preset_name",
            on_change=handle_preset_change,
            horizontal=True,
        )

        selected_models = st.multiselect(
            "模型",
            options=model_options,
            default=current_models,
            key="selected_models",
        )

        metric_options = list(iter_selectable_metric_keys())
        selected_metric = st.selectbox(
            "指标",
            options=metric_options,
            format_func=selectable_metric_label,
            key="selected_metric",
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
            format_func=lambda value: "不限" if value is None else f"<= {float(value):.2f}",
            key="enrol_quality_max",
        )
        st.selectbox(
            "enrol length (GT词数)",
            options=ENROL_GT_LENGTH_OPTIONS,
            format_func=lambda value: ENROL_GT_LENGTH_LABELS.get(value, str(value)),
            key="enrol_gt_length_filter",
        )

        st.divider()
        st.subheader("Refresh")
        auto_refresh_label = st.selectbox(
            "自动刷新",
            options=list(AUTO_REFRESH_OPTIONS.keys()),
            key="auto_refresh_label",
        )
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Refresh now", width="stretch"):
                load_dashboard_state.cache_clear()
                st.rerun()
        with col_right:
            if st.button("Clear cache", width="stretch"):
                load_dashboard_state.cache_clear()
                st.rerun()

    return selected_models, selected_metric, AUTO_REFRESH_OPTIONS[auto_refresh_label]


def main() -> None:
    ensure_session_defaults()
    selected_refresh_seconds = AUTO_REFRESH_OPTIONS.get(st.session_state["auto_refresh_label"], 0)
    refresh_token = int(time.time() // selected_refresh_seconds) if selected_refresh_seconds > 0 else 0
    state = load_dashboard_state(refresh_token)
    selected_models, selected_metric, auto_refresh_seconds = render_sidebar(state.models)
    schedule_auto_refresh(auto_refresh_seconds)

    st.title("REAL-T BASE Dashboard")
    st.caption(f"Source root: `{REPO_ROOT / 'output' / 'BASE'}`")

    if state.transcript_length_note:
        st.warning(state.transcript_length_note)

    if not state.models:
        st.error("No model directories were found under `output/BASE`.")
        st.stop()

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

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

    subset_label = (
        f"Custom subset (based on {st.session_state['preset_name']})"
        if is_custom_subset(st.session_state["preset_name"], filter_config)
        else st.session_state["preset_name"]
    )
    summary_items = [
        f"Subset: {subset_label}",
        f"Speaker ratio >= {filter_config.speaker_ratio_min}%",
        f"Transcript length > {filter_config.transcript_length_min}",
        f"Speaker scope: {SPEAKER_SCOPE_LABELS[filter_config.speaker_scope]}",
        f"Mixture ratio: {format_optional_threshold(filter_config.mixture_ratio_min, '>= ')}%"
        if filter_config.mixture_ratio_min is not None
        else "Mixture ratio: 不限",
        f"Mixture duration: {format_optional_threshold(filter_config.mixture_duration_max, '<= ')}s"
        if filter_config.mixture_duration_max is not None
        else "Mixture duration: 不限",
    ]
    st.info(" | ".join(summary_items))

    filtered_rows = filter_metric_rows(
        metric_long_df=state.metric_long_df,
        selected_models=selected_models,
        metric_key=selected_metric,
        filter_config=filter_config,
    )
    aggregates = aggregate_metric_rows(
        filtered_metric_rows=filtered_rows,
        selected_models=selected_models,
        metric_key=selected_metric,
        availability_df=state.availability_df,
    )

    metric_label = selectable_metric_label(selected_metric)
    st.subheader(metric_label)

    if is_sim_ui_metric(selected_metric):
        overall_display_df = format_sim_overall_table(aggregates["overall_df"])
    else:
        overall_display_df = format_overall_table(aggregates["overall_df"])
    st.markdown("**Overall**")
    st.dataframe(overall_display_df, width="stretch", hide_index=True)

    st.markdown("**By Dataset**")
    if is_sim_ui_metric(selected_metric):
        ds_df = format_sim_group_summary_table(aggregates["dataset_summary_df"])
    else:
        ds_df = format_group_summary_table(aggregates["dataset_summary_df"])
    st.dataframe(ds_df, width="stretch", hide_index=True)

    st.markdown("**By Language**")
    if is_sim_ui_metric(selected_metric):
        lang_df = format_sim_group_summary_table(aggregates["language_summary_df"])
    else:
        lang_df = format_group_summary_table(aggregates["language_summary_df"])
    st.dataframe(lang_df, width="stretch", hide_index=True)

    chart_left, chart_right = st.columns([1.3, 1.0])
    with chart_left:
        st.markdown("**Mean Heatmap**")
        ter_metric = (
            not is_sim_ui_metric(selected_metric)
            and METRIC_SPECS[selected_metric]["source_type"] == "ter"
        )
        sim_mode = is_sim_ui_metric(selected_metric)
        st.plotly_chart(
            build_heatmap(
                aggregates["heatmap_df"],
                lower_is_better=ter_metric,
                sim_uplift=sim_mode,
            ),
            width="stretch",
        )

    group_labels = ["Overall", *DATASET_ORDER, *LANGUAGE_ORDER]
    with chart_right:
        if st.session_state["bar_group_label"] not in group_labels:
            st.session_state["bar_group_label"] = "Overall"
        st.selectbox("柱状图分组", options=group_labels, key="bar_group_label")
        st.plotly_chart(
            build_group_bar_chart(
                aggregates["summary_long_df"],
                st.session_state["bar_group_label"],
                sim_uplift=is_sim_ui_metric(selected_metric),
            ),
            width="stretch",
        )

    with st.expander("Metric File Status", expanded=False):
        if is_sim_ui_metric(selected_metric):
            status_mask = state.availability_df["metric_key"].isin(SIM_METRIC_KEYS)
        else:
            status_mask = state.availability_df["metric_key"] == selected_metric
        status_df = state.availability_df[status_mask].copy()
        status_df = status_df[status_df["model"].isin(selected_models)].copy()
        status_df = status_df[
            ["model", "status_label", "row_count", "matched_row_count", "file_path", "message"]
        ].rename(
            columns={
                "status_label": "status",
                "row_count": "csv_rows",
                "matched_row_count": "matched_rows",
                "file_path": "file",
            }
        )
        st.dataframe(status_df, width="stretch", hide_index=True)

    if is_sim_ui_metric(selected_metric):
        st.caption(
            "SIM：表格中 `SIM` 列为 enrol-mixture 与 enrol-tse 的组内均值及相对变化 "
            "((mean_tse - mean_mix) / mean_mix × 100%，带 ↑/↓)。"
            "热力图与柱状图仅展示该 **Δ%**（越大表示 tse 相对 mixture 提升越多）。"
            "`count` 为同时有两种 SIM 的配对 `utterance_key` 数。"
        )
    st.caption(
        "Column `count` = number of distinct `utterance_key` in that slice. "
        "For BASE/TER, `utterance_key` is built from mixture + enrolment utterance, "
        "so the same `mixture_utterance` with different targets/enrol rows counts as separate rows — "
        "this is not deduplication by mixture alone. "
        "Model columns are the mean of `metric_value`. "
        "Overall still shows per-model CSV row counts, which can differ by model."
    )
    st.caption(f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
