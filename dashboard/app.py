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
        LANGUAGE_ORDER,
        METRIC_SPECS,
        SIM_METRIC_KEYS,
        aggregate_metric_rows,
        build_filter_config,
        clear_dashboard_caches,
        filter_metric_rows,
        format_group_summary_table,
        format_overall_table,
        format_sim_group_summary_table,
        format_sim_overall_table,
        is_custom_subset,
        is_sim_ui_metric,
        iter_selectable_metric_keys,
        load_dashboard_catalog,
        load_effective_metric_view,
        selectable_metric_label,
    )
    from dashboard.samples_page import clear_samples_page_runtime_caches, render_samples_page
    from dashboard.ui_shared import (
        AUTO_REFRESH_OPTIONS,
        ENROL_GT_LENGTH_LABELS,
        ENROL_GT_LENGTH_OPTIONS,
        ENROL_QUALITY_OPTIONS,
        MIXTURE_DURATION_OPTIONS,
        MIXTURE_RATIO_OPTIONS,
        PAGE_REFS,
        SPEAKER_COUNT_OPTIONS,
        SPEAKER_RATIO_OPTIONS,
        SPEAKER_SCOPE_LABELS,
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
        LANGUAGE_ORDER,
        METRIC_SPECS,
        SIM_METRIC_KEYS,
        aggregate_metric_rows,
        build_filter_config,
        clear_dashboard_caches,
        filter_metric_rows,
        format_group_summary_table,
        format_overall_table,
        format_sim_group_summary_table,
        format_sim_overall_table,
        is_custom_subset,
        is_sim_ui_metric,
        iter_selectable_metric_keys,
        load_dashboard_catalog,
        load_effective_metric_view,
        selectable_metric_label,
    )
    from samples_page import clear_samples_page_runtime_caches, render_samples_page
    from ui_shared import (
        AUTO_REFRESH_OPTIONS,
        ENROL_GT_LENGTH_LABELS,
        ENROL_GT_LENGTH_OPTIONS,
        ENROL_QUALITY_OPTIONS,
        MIXTURE_DURATION_OPTIONS,
        MIXTURE_RATIO_OPTIONS,
        PAGE_REFS,
        SPEAKER_COUNT_OPTIONS,
        SPEAKER_RATIO_OPTIONS,
        SPEAKER_SCOPE_LABELS,
        SPEAKER_SCOPE_OPTIONS,
        TRANSCRIPT_LENGTH_OPTIONS,
        build_filter_summary_items,
        default_model_selection,
        ensure_session_defaults,
        handle_preset_change,
        render_page_header,
    )


st.set_page_config(
    page_title="REAL-T Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


def wrap_model_display_name(value: str) -> str:
    return value.replace("_", "\n_")


def build_table_column_config(df: pd.DataFrame, data_column_start: int = 2) -> dict[str, object]:
    config: dict[str, object] = {}
    for idx, column in enumerate(df.columns):
        width = "medium" if idx >= data_column_start else "small"
        config[str(column)] = st.column_config.TextColumn(str(column), width=width)
    return config


def inject_dataframe_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] [role="columnheader"] div,
        [data-testid="stDataFrame"] [role="gridcell"] div {
            white-space: pre-wrap !important;
            line-height: 1.15 !important;
            word-break: break-word;
        }
        [data-testid="stDataFrame"] [role="columnheader"] {
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
            lambda value: "" if pd.isna(value) else f"{float(value):+.4f}%",
        )
        hover_z = "%{z:+.4f}%"
        cbar_title = "Δ% (↑ better)"
        cbar_fmt: dict[str, str] = {"tickformat": "+.4f", "ticksuffix": "%"}
        reverse = False
    else:
        labels = matrix.map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        hover_z = "%{z:.4f}"
        cbar_title = (
            "Mean (lower is better)" if lower_is_better else "Mean (higher is better)"
        )
        cbar_fmt = {"tickformat": ".4f"}
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
        text_vals = chart_df["mean"].map(lambda value: f"{value:+.4f}%")
        hover_y = "%{y:+.4f}%"
    else:
        text_vals = chart_df["mean"].map(lambda value: f"{value:.4f}")
        hover_y = "%{y:.4f}"

    figure.add_bar(
        x=chart_df["model"],
        y=chart_df["mean"],
        text=text_vals,
        textposition="outside",
        marker_color="#2962ff",
        hovertemplate=f"%{{x}}<br>{hover_y}<extra></extra>",
    )
    yaxis_cfg = (
        {"tickformat": "+.4f", "ticksuffix": "%"} if sim_uplift else {"tickformat": ".4f"}
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


@st.cache_resource(show_spinner=False)
def get_dashboard_catalog_cached(refresh_token: int) -> object:
    return load_dashboard_catalog(refresh_token)


@st.cache_resource(show_spinner=False)
def get_metric_view_cached(
    preset_name: str,
    metric_key: str,
    selected_models: tuple[str, ...],
    refresh_token: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_effective_metric_view(preset_name, metric_key, selected_models, refresh_token)


def render_sidebar(model_options: list[str]) -> tuple[list[str], str, int]:
    stored_models = st.session_state.get("selected_models")
    if stored_models is None:
        current_models = default_model_selection(model_options)
    else:
        current_models = [m for m in stored_models if m in model_options]
        if not current_models:
            current_models = default_model_selection(model_options)

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
            format_func=lambda value: "不限" if value is None else f"<= {float(value):.4f}",
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
            if st.button("Refresh now"):
                clear_dashboard_caches()
                get_dashboard_catalog_cached.clear()
                get_metric_view_cached.clear()
                clear_samples_page_runtime_caches()
                st.rerun()
        with col_right:
            if st.button("Clear cache"):
                clear_dashboard_caches()
                get_dashboard_catalog_cached.clear()
                get_metric_view_cached.clear()
                clear_samples_page_runtime_caches()
                st.rerun()

    return selected_models, selected_metric, AUTO_REFRESH_OPTIONS[auto_refresh_label]


def main() -> None:
    ensure_session_defaults()
    inject_dataframe_styles()
    selected_refresh_seconds = AUTO_REFRESH_OPTIONS.get(st.session_state["auto_refresh_label"], 0)
    refresh_token = int(time.time() // selected_refresh_seconds) if selected_refresh_seconds > 0 else 0
    catalog = get_dashboard_catalog_cached(refresh_token)
    selected_models, selected_metric, auto_refresh_seconds = render_sidebar(catalog.models)
    schedule_auto_refresh(auto_refresh_seconds)

    render_page_header("REAL-T Dashboard", "Samples", "samples")
    st.caption(f"BASE results root: `{REPO_ROOT / 'output' / 'BASE'}`")
    st.caption(f"PRIMARY results root: `{REPO_ROOT / 'output' / 'PRIMARY'}`")
    st.caption(
        f"PRIMARY logic source: `{REPO_ROOT / 'datasets' / 'REAL-T' / 'metadata'}`"
    )
    st.caption(
        "PRIMARY 预设会优先使用 `output/BASE`；只有当某模型不存在于 BASE 时，才回退到 `output/PRIMARY`。"
    )
    if catalog.enrol_quality_source_path:
        st.caption(f"Enrol quality source: `{catalog.enrol_quality_source_path}`")

    if catalog.transcript_length_note:
        st.warning(catalog.transcript_length_note)
    if catalog.enrol_quality_note:
        st.warning(catalog.enrol_quality_note)

    if not catalog.models:
        st.error("No model directories were found under `output/BASE` or `output/PRIMARY`.")
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
    effective_metric_long_df, effective_availability_df = get_metric_view_cached(
        filter_config.preset_name,
        selected_metric,
        tuple(selected_models),
        refresh_token,
    )

    summary_items = build_filter_summary_items(
        filter_config,
        st.session_state["preset_name"],
        custom_subset=is_custom_subset(st.session_state["preset_name"], filter_config),
    )
    st.info(" | ".join(summary_items))

    filtered_rows = filter_metric_rows(
        metric_long_df=effective_metric_long_df,
        selected_models=selected_models,
        metric_key=selected_metric,
        filter_config=filter_config,
    )
    aggregates = aggregate_metric_rows(
        filtered_metric_rows=filtered_rows,
        selected_models=selected_models,
        metric_key=selected_metric,
        availability_df=effective_availability_df,
    )

    metric_label = selectable_metric_label(selected_metric)
    st.subheader(metric_label)

    if is_sim_ui_metric(selected_metric):
        overall_display_df = format_sim_overall_table(aggregates["overall_df"])
    else:
        overall_display_df = format_overall_table(aggregates["overall_df"])
    st.markdown("**Overall**")
    st.dataframe(
        overall_display_df,
        use_container_width=True,
        hide_index=True,
        column_config=build_table_column_config(overall_display_df, data_column_start=0),
    )

    st.markdown("**By Dataset**")
    if is_sim_ui_metric(selected_metric):
        ds_df = format_sim_group_summary_table(aggregates["dataset_summary_df"])
    else:
        ds_df = format_group_summary_table(aggregates["dataset_summary_df"])
    st.dataframe(
        ds_df,
        use_container_width=True,
        hide_index=True,
        column_config=build_table_column_config(ds_df, data_column_start=2),
    )

    st.markdown("**By Language**")
    if is_sim_ui_metric(selected_metric):
        lang_df = format_sim_group_summary_table(aggregates["language_summary_df"])
    else:
        lang_df = format_group_summary_table(aggregates["language_summary_df"])
    st.dataframe(
        lang_df,
        use_container_width=True,
        hide_index=True,
        column_config=build_table_column_config(lang_df, data_column_start=2),
    )

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
            use_container_width=True,
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
            use_container_width=True,
        )

    with st.expander("Metric File Status", expanded=False):
        if is_sim_ui_metric(selected_metric):
            status_mask = effective_availability_df["metric_key"].isin(SIM_METRIC_KEYS)
        else:
            status_mask = effective_availability_df["metric_key"] == selected_metric
        status_df = effective_availability_df[status_mask].copy()
        status_df = status_df[status_df["model"].isin(selected_models)].copy()
        status_df["model"] = status_df["model"].map(wrap_model_display_name)
        status_df = status_df[
            [
                "model",
                "result_root",
                "status_label",
                "row_count",
                "matched_row_count",
                "file_path",
                "message",
            ]
        ].rename(
            columns={
                "result_root": "source",
                "status_label": "status",
                "row_count": "csv_rows",
                "matched_row_count": "matched_rows",
                "file_path": "file",
            }
        )
        st.dataframe(
            status_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "model": st.column_config.TextColumn("model", width="small"),
                "source": st.column_config.TextColumn("source", width="small"),
                "status": st.column_config.TextColumn("status", width="small"),
                "csv_rows": st.column_config.TextColumn("csv_rows", width="small"),
                "matched_rows": st.column_config.TextColumn("matched_rows", width="small"),
                "file": st.column_config.TextColumn("file", width="large"),
                "message": st.column_config.TextColumn("message", width="large"),
            },
        )

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


def run_app() -> None:
    dashboard_page = st.Page(main, title="Dashboard", icon="📊", default=True)
    samples_page = st.Page(render_samples_page, title="Samples", icon="🎧", url_path="samples")
    PAGE_REFS["dashboard"] = dashboard_page
    PAGE_REFS["samples"] = samples_page
    navigation = st.navigation([dashboard_page, samples_page], position="hidden")
    navigation.run()


if __name__ == "__main__":
    run_app()
