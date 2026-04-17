"""Dash application factory for the connectivity analysis dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from dash import Dash, Input, Output, State, dcc, html

from data_vis_py.io.dataset_loader import DatasetBundle, load_dataset
from data_vis_py.stats.analysis import (
    ALL_GROUPS_LABEL,
    ALL_GROUPS_VALUE,
    AnalysisConfig,
    SELECTED_EDGE_DELTA_LABEL,
    SELECTED_EDGE_DELTA_VALUE,
    run_analysis,
    run_covariate_analysis,
    run_multivariate_regression_analysis,
    summarize_pair_result,
)


DISPLAY_OPTIONS = {
    "effect": ("Effect (Mean Delta)", "matrix"),
    "p": ("P-Value", "p_matrix"),
    "q": ("Q-Value", "q_matrix"),
}

BAND_PRESETS = {
    "custom": None,
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def create_dashboard(
    dataset_dir,
    initial_bundle: DatasetBundle,
    json_files: list[str],
    csv_files: list[str],
    initial_json: str,
    initial_csv: str,
) -> Dash:
    """Create the Dash app for interactive connectivity analysis."""
    app = Dash(__name__)
    app.title = "MEG Connectivity Explorer"

    numeric_subject_columns = [
        column
        for column in initial_bundle.subjects.columns
        if column not in {"dataset_id", "subject_id", "base_subject_id", "group_label", "mtime"}
        and str(initial_bundle.subjects[column].dtype) != "object"
    ]
    group_options = sorted(initial_bundle.subjects["group_label"].dropna().astype(str).unique().tolist())
    default_pair = f"{initial_bundle.channels[0]}|{initial_bundle.channels[1]}"

    app.layout = html.Div(
        [
            dcc.Store(id="selected-pair", data=default_pair),
            html.Div(
                [
                    html.H2("MEG Connectivity Explorer"),
                    html.P("Interactive paired contrasts and covariate analysis for REST_24_Stroke."),
                    _sidebar(initial_bundle, numeric_subject_columns, group_options, json_files, csv_files, initial_json, initial_csv),
                    html.Div(id="analysis-summary", className="panel"),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="Heatmap",
                                children=[
                                    html.Div(id="heatmap-description", className="panel"),
                                    dcc.Graph(id="heatmap"),
                                    dcc.Graph(id="subject-bar-chart"),
                                    html.Div(id="subject-bar-stats", className="panel"),
                                    html.Details(
                                        [
                                            html.Summary("Correlation analysis"),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Correlation variable"),
                                                            dcc.Dropdown(
                                                                id="heatmap-correlation-variable",
                                                                options=[{"label": column, "value": column} for column in numeric_subject_columns],
                                                                value=numeric_subject_columns[0] if numeric_subject_columns else None,
                                                                clearable=True,
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                    dcc.Graph(id="heatmap-correlation-chart"),
                                                    html.Div(id="heatmap-correlation-stats", className="panel"),
                                                ],
                                                style={"marginTop": "10px"},
                                            ),
                                        ],
                                        open=False,
                                        className="panel",
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Multivariate regression analysis"),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Regression covariates"),
                                                            dcc.Dropdown(
                                                                id="regression-covariates",
                                                                options=[{"label": SELECTED_EDGE_DELTA_LABEL, "value": SELECTED_EDGE_DELTA_VALUE}]
                                                                + [{"label": column, "value": column} for column in numeric_subject_columns],
                                                                value=numeric_subject_columns[:2],
                                                                multi=True,
                                                            ),
                                                            html.Label("Primary outcome variable"),
                                                            dcc.Dropdown(
                                                                id="heatmap-regression-outcome-variable",
                                                                options=[{"label": SELECTED_EDGE_DELTA_LABEL, "value": SELECTED_EDGE_DELTA_VALUE}]
                                                                + [{"label": column, "value": column} for column in numeric_subject_columns],
                                                                value=SELECTED_EDGE_DELTA_VALUE,
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                    html.Div(id="heatmap-regression-summary", className="panel"),
                                                    dcc.Graph(id="heatmap-regression-fit-chart"),
                                                    dcc.Graph(id="heatmap-regression-residual-chart"),
                                                ],
                                                style={"marginTop": "10px"},
                                            ),
                                        ],
                                        open=False,
                                        className="panel",
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Detail",
                                children=[
                                    html.Div(id="pair-summary", className="panel"),
                                    dcc.Graph(id="pair-detail-chart"),
                                ],
                            ),
                            dcc.Tab(
                                label="Covariates",
                                children=[
                                    html.Div(id="covariate-summary", className="panel"),
                                    dcc.Graph(id="covariate-chart"),
                                ],
                            ),
                        ]
                    )
                ],
                className="main",
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "320px minmax(0, 1fr)",
            "gap": "16px",
            "padding": "16px",
            "fontFamily": "Helvetica, Arial, sans-serif",
        },
    )

    def current_bundle(json_filename: str, csv_filename: str) -> DatasetBundle:
        return load_dataset(dataset_dir, json_filename=json_filename, csv_filename=csv_filename)

    @app.callback(
        Output("group-a", "options"),
        Output("group-a", "value"),
        Output("group-b", "options"),
        Output("group-b", "value"),
        Output("correlation-variable", "options"),
        Output("correlation-variable", "value"),
        Output("heatmap-correlation-variable", "options"),
        Output("heatmap-correlation-variable", "value"),
        Output("regression-covariates", "options"),
        Output("regression-covariates", "value"),
        Output("heatmap-regression-outcome-variable", "options"),
        Output("heatmap-regression-outcome-variable", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
    )
    def sync_file_dependent_controls(json_filename: str, csv_filename: str):
        bundle = current_bundle(json_filename, csv_filename)
        groups = sorted(bundle.subjects["group_label"].dropna().astype(str).unique().tolist())
        group_options = [{"label": ALL_GROUPS_LABEL, "value": ALL_GROUPS_VALUE}] + [{"label": group, "value": group} for group in groups]
        numeric_columns = [
            column
            for column in bundle.subjects.columns
            if column not in {"dataset_id", "subject_id", "base_subject_id", "group_label", "mtime"}
            and str(bundle.subjects[column].dtype) != "object"
        ]
        numeric_options = [{"label": column, "value": column} for column in numeric_columns]
        regression_variable_options = [{"label": SELECTED_EDGE_DELTA_LABEL, "value": SELECTED_EDGE_DELTA_VALUE}] + numeric_options
        default_group = ALL_GROUPS_VALUE if groups else None
        default_corr = numeric_columns[0] if numeric_columns else None
        default_regression = numeric_columns[:2]
        return (
            group_options,
            default_group,
            group_options,
            default_group,
            numeric_options,
            default_corr,
            numeric_options,
            default_corr,
            regression_variable_options,
            default_regression,
            regression_variable_options,
            SELECTED_EDGE_DELTA_VALUE,
        )

    @app.callback(
        Output("freq-min", "value"),
        Output("freq-max", "value"),
        Input("band-preset", "value"),
        State("freq-min", "value"),
        State("freq-max", "value"),
    )
    def sync_band_preset(band_preset: str, current_min: float, current_max: float) -> tuple[float, float]:
        preset = BAND_PRESETS.get(band_preset)
        if preset is None:
            return current_min, current_max
        return preset

    @app.callback(
        Output("longitudinal-value-a", "options"),
        Output("longitudinal-value-b", "options"),
        Output("longitudinal-value-a", "value"),
        Output("longitudinal-value-b", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        Input("longitudinal-enabled", "value"),
    )
    def sync_longitudinal_value_options(json_filename: str, csv_filename: str, enabled_flags: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
        if "enabled" not in (enabled_flags or []):
            return [], [], None, None
        bundle = current_bundle(json_filename, csv_filename)
        values = [value for value in bundle.subjects["mtime"].dropna().unique().tolist()]
        values = sorted(values, key=lambda item: str(item))
        options = [{"label": str(value), "value": value} for value in values]
        first = values[0] if values else None
        second = values[1] if len(values) > 1 else first
        return options, options, first, second

    @app.callback(
        Output("selected-pair", "data"),
        Input("heatmap", "clickData"),
        State("selected-pair", "data"),
        prevent_initial_call=True,
    )
    def select_pair(click_data: dict[str, Any] | None, current_pair: str) -> str:
        if not click_data or not click_data.get("points"):
            return current_pair
        point = click_data["points"][0]
        roi_from = point.get("y")
        roi_to = point.get("x")
        if roi_from == roi_to:
            return current_pair
        bundle = initial_bundle
        ordered = sorted([roi_from, roi_to], key=bundle.channels.index)
        return f"{ordered[0]}|{ordered[1]}"

    @app.callback(
        Output("analysis-summary", "children"),
        Output("heatmap-description", "children"),
        Output("heatmap", "figure"),
        Output("subject-bar-chart", "figure"),
        Output("subject-bar-stats", "children"),
        Output("heatmap-correlation-chart", "figure"),
        Output("heatmap-correlation-stats", "children"),
        Output("heatmap-regression-summary", "children"),
        Output("heatmap-regression-fit-chart", "figure"),
        Output("heatmap-regression-residual-chart", "figure"),
        Output("pair-summary", "children"),
        Output("pair-detail-chart", "figure"),
        Output("covariate-summary", "children"),
        Output("covariate-chart", "figure"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        Input("group-a", "value"),
        Input("group-b", "value"),
        Input("longitudinal-enabled", "value"),
        Input("longitudinal-require-pairs", "value"),
        Input("longitudinal-value-a", "value"),
        Input("longitudinal-value-b", "value"),
        Input("trial-a", "value"),
        Input("trial-b", "value"),
        Input("band-preset", "value"),
        Input("freq-min", "value"),
        Input("freq-max", "value"),
        Input("mtime-filter", "value"),
        Input("correction-method", "value"),
        Input("display-mode", "value"),
        Input("significance-threshold", "value"),
        Input("correlation-method", "value"),
        Input("correlation-variable", "value"),
        Input("heatmap-correlation-variable", "value"),
        Input("regression-covariates", "value"),
        Input("heatmap-regression-outcome-variable", "value"),
        Input("selected-pair", "data"),
    )
    def update_dashboard(
        json_filename: str,
        csv_filename: str,
        group_a: str,
        group_b: str,
        longitudinal_enabled_flags: list[str],
        longitudinal_pairing_flags: list[str],
        longitudinal_value_a: Any,
        longitudinal_value_b: Any,
        trial_a: int,
        trial_b: int,
        band_preset: str,
        freq_min: float,
        freq_max: float,
        mtime_filter: str,
        correction_method: str,
        display_mode: str,
        significance_threshold: float,
        correlation_method: str,
        correlation_variable: str | None,
        heatmap_correlation_variable: str | None,
        regression_covariates: list[str] | None,
        heatmap_regression_outcome_variable: str | None,
        selected_pair: str,
    ) -> tuple[Any, Any, go.Figure, go.Figure, Any, go.Figure, Any, Any, go.Figure, go.Figure, Any, go.Figure, Any, go.Figure]:
        bundle = current_bundle(json_filename, csv_filename)
        longitudinal_enabled = "enabled" in (longitudinal_enabled_flags or [])
        longitudinal_require_pairs = "paired_only" in (longitudinal_pairing_flags or [])
        if BAND_PRESETS.get(band_preset):
            freq_min, freq_max = BAND_PRESETS[band_preset]
        freq_min = float(freq_min)
        freq_max = float(freq_max)
        if freq_min > freq_max:
            freq_min, freq_max = freq_max, freq_min
        significance_threshold = float(significance_threshold) if significance_threshold is not None else 0.05
        significance_threshold = min(max(significance_threshold, 0.0), 1.0)
        config = AnalysisConfig(
            dataset_id=bundle.dataset_id,
            metric=bundle.metric,
            trial_a=int(trial_a),
            trial_b=int(trial_b),
            freq_min=freq_min,
            freq_max=freq_max,
            mtime_filter=mtime_filter,
            correction_method=correction_method,
            group_a=group_a,
            group_b=group_b,
            longitudinal_enabled=longitudinal_enabled,
            longitudinal_require_pairs=longitudinal_require_pairs,
            longitudinal_column="mtime",
            longitudinal_value_a=longitudinal_value_a,
            longitudinal_value_b=longitudinal_value_b,
        )
        analysis_result = run_analysis(bundle, config)
        effective_analysis_mode = "longitudinal_delta" if longitudinal_enabled else "trial_delta"
        pair_summary = summarize_pair_result(analysis_result, selected_pair)
        covariate_result = run_covariate_analysis(
            bundle,
            analysis_result,
            selected_pair,
            correlation_method=correlation_method,
            correlation_variable=correlation_variable,
            regression_covariates=regression_covariates,
        )
        heatmap_correlation_result = run_covariate_analysis(
            bundle,
            analysis_result,
            selected_pair,
            correlation_method=correlation_method,
            correlation_variable=heatmap_correlation_variable,
            regression_covariates=None,
        )
        heatmap_regression_result = run_multivariate_regression_analysis(
            bundle,
            analysis_result,
            selected_pair,
            outcome_variable=heatmap_regression_outcome_variable,
            regression_covariates=regression_covariates,
        )

        summary_component = html.Div(
            [
                html.H4(analysis_result["analysis_label"]),
                html.P(f"Dataset: {bundle.dataset_id}"),
                html.P(f"Metric: {bundle.metric}"),
                html.P(f"Frequency band: {analysis_result['freq_label']}"),
                html.P(f"MTime filter: {mtime_filter}"),
                html.P(f"Correction: {analysis_result['correction_label']}"),
                html.P(
                    f"Groups: {analysis_result['selected_group_a']} vs {analysis_result['selected_group_b']}"
                    if analysis_result["between_groups"]
                    else f"Group: {analysis_result['selected_group_a']}"
                ),
                html.P(
                    f"Longitudinal mode: MTime {longitudinal_value_a} vs {longitudinal_value_b} | "
                    f"{'paired repeated-subject analysis only' if longitudinal_require_pairs else 'all available subjects at each selected timepoint'}"
                    if longitudinal_enabled
                    else "Longitudinal mode: off"
                ),
            ]
        )

        heatmap_description = _build_heatmap_description(analysis_result, display_mode, mtime_filter, significance_threshold)

        heatmap_figure = _build_heatmap(
            analysis_result=analysis_result,
            display_mode=display_mode,
            selected_pair=selected_pair,
            significance_threshold=significance_threshold,
        )
        subject_bar_figure = _build_subject_bar_chart(pair_summary, effective_analysis_mode)
        subject_bar_stats = _build_subject_bar_stats(pair_summary)
        heatmap_corr_figure, heatmap_corr_stats = _build_heatmap_correlation_view(
            heatmap_correlation_result,
            heatmap_correlation_variable,
        )
        heatmap_regression_summary, heatmap_regression_fit, heatmap_regression_residual = _build_heatmap_regression_view(
            heatmap_regression_result,
        )

        pair_component, pair_figure = _build_detail_view(pair_summary, effective_analysis_mode)
        cov_component, cov_figure = _build_covariate_view(covariate_result, effective_analysis_mode, correlation_variable)
        return (
            summary_component,
            heatmap_description,
            heatmap_figure,
            subject_bar_figure,
            subject_bar_stats,
            heatmap_corr_figure,
            heatmap_corr_stats,
            heatmap_regression_summary,
            heatmap_regression_fit,
            heatmap_regression_residual,
            pair_component,
            pair_figure,
            cov_component,
            cov_figure,
        )

    return app


def _sidebar(
    bundle: DatasetBundle,
    numeric_subject_columns: list[str],
    group_options: list[str],
    json_files: list[str],
    csv_files: list[str],
    initial_json: str,
    initial_csv: str,
) -> html.Div:
    trial_options = [{"label": str(trial_id), "value": trial_id} for trial_id in bundle.trial_ids]
    mtime_options = [{"label": "All", "value": "All"}] + [
        {"label": label, "value": label} for label in sorted(bundle.subjects["mtime"].dropna().unique())
    ]
    positive_frequencies = [frequency for frequency in bundle.frequencies if frequency > 0]
    group_dropdown_options = [{"label": ALL_GROUPS_LABEL, "value": ALL_GROUPS_VALUE}] + [
        {"label": group, "value": group} for group in group_options
    ]

    return html.Div(
        [
            html.Label("Dataset"),
            dcc.Dropdown(options=[{"label": bundle.dataset_id, "value": bundle.dataset_id}], value=bundle.dataset_id, disabled=True),
            html.Label("JSON file"),
            dcc.Dropdown(
                id="json-file",
                options=[{"label": name, "value": name} for name in json_files],
                value=initial_json,
                clearable=False,
            ),
            html.Label("CSV file"),
            dcc.Dropdown(
                id="csv-file",
                options=[{"label": name, "value": name} for name in csv_files],
                value=initial_csv,
                clearable=False,
            ),
            html.Label("Metric"),
            dcc.Dropdown(options=[{"label": bundle.metric, "value": bundle.metric}], value=bundle.metric, disabled=True),
            html.Label("Trial A"),
            dcc.Dropdown(id="trial-a", options=trial_options, value=bundle.trial_ids[0], clearable=False),
            html.Label("Trial B"),
            dcc.Dropdown(id="trial-b", options=trial_options, value=bundle.trial_ids[1], clearable=False),
            html.Label("Group 1"),
            dcc.Dropdown(
                id="group-a",
                options=group_dropdown_options,
                value=ALL_GROUPS_VALUE if group_options else None,
                clearable=False,
            ),
            html.Label("Group 2"),
            dcc.Dropdown(
                id="group-b",
                options=group_dropdown_options,
                value=ALL_GROUPS_VALUE if group_options else None,
                clearable=False,
            ),
            html.Label("Longitudinal analysis"),
            dcc.Checklist(
                id="longitudinal-enabled",
                options=[{"label": "Enable longitudinal delta-vs-delta analysis", "value": "enabled"}],
                value=[],
            ),
            html.Label("Longitudinal pairing"),
            dcc.Checklist(
                id="longitudinal-require-pairs",
                options=[{"label": "Only include subjects measured at both selected timepoints", "value": "paired_only"}],
                value=["paired_only"],
            ),
            html.Label("Timepoint 1"),
            dcc.Dropdown(id="longitudinal-value-a", options=[], value=None, clearable=False),
            html.Label("Timepoint 2"),
            dcc.Dropdown(id="longitudinal-value-b", options=[], value=None, clearable=False),
            html.Label("Band preset"),
            dcc.RadioItems(
                id="band-preset",
                options=[
                    {"label": "Custom", "value": "custom"},
                    {"label": "Delta", "value": "delta"},
                    {"label": "Theta", "value": "theta"},
                    {"label": "Alpha", "value": "alpha"},
                    {"label": "Beta", "value": "beta"},
                    {"label": "Gamma", "value": "gamma"},
                ],
                value="alpha",
            ),
            html.Label("Frequency from (Hz)"),
            dcc.Input(id="freq-min", type="number", value=8.0, step=0.1, min=min(positive_frequencies), max=max(positive_frequencies)),
            html.Label("Frequency to (Hz)"),
            dcc.Input(id="freq-max", type="number", value=13.0, step=0.1, min=min(positive_frequencies), max=max(positive_frequencies)),
            html.Label("MTime filter"),
            dcc.Dropdown(id="mtime-filter", options=mtime_options, value="All", clearable=False),
            html.Label("Multiple testing"),
            dcc.Dropdown(
                id="correction-method",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "FDR-BH", "value": "fdr_bh"},
                    {"label": "Bonferroni", "value": "bonferroni"},
                    {"label": "Holm", "value": "holm"},
                ],
                value="fdr_bh",
                clearable=False,
            ),
            html.Label("Heatmap mode"),
            dcc.RadioItems(
                id="display-mode",
                options=[{"label": label, "value": key} for key, (label, _) in DISPLAY_OPTIONS.items()],
                value="p",
            ),
            html.Label("Significance threshold"),
            dcc.Input(id="significance-threshold", type="number", value=0.05, step=0.001, min=0.0, max=1.0),
            html.Hr(),
            html.Label("Correlation method"),
            dcc.RadioItems(
                id="correlation-method",
                options=[
                    {"label": "Pearson", "value": "pearson"},
                    {"label": "Spearman", "value": "spearman"},
                ],
                value="pearson",
            ),
            html.Label("Correlation variable"),
            dcc.Dropdown(
                id="correlation-variable",
                options=[{"label": column, "value": column} for column in numeric_subject_columns],
                value=numeric_subject_columns[0] if numeric_subject_columns else None,
                clearable=True,
            ),
        ],
        style={"display": "grid", "gap": "8px"},
    )


def _build_heatmap(
    analysis_result: dict[str, Any],
    display_mode: str,
    selected_pair: str,
    significance_threshold: float,
) -> go.Figure:
    display_label, key = DISPLAY_OPTIONS[display_mode]
    z_values = np.array(analysis_result[key], dtype=float)
    pair_text = np.empty_like(z_values, dtype=object)
    value_text = np.empty_like(z_values, dtype=object)
    for i, roi_from in enumerate(analysis_result["roi_order"]):
        for j, roi_to in enumerate(analysis_result["roi_order"]):
            pair_text[i, j] = f"{roi_from} -> {roi_to}"
            value = z_values[i, j]
            value_text[i, j] = "" if np.isnan(value) else f"{value:.3f}"

    colorscale: str | list[list[float | str]] = "RdBu_r" if display_mode == "effect" else "Viridis"
    zmid: float | None = 0 if display_mode == "effect" else None
    zmin: float | None = None
    zmax: float | None = None
    if display_mode == "p":
        split = min(max(significance_threshold, 0.0), 1.0)
        lower_1 = max(split * 0.4, 0.0)
        lower_2 = max(split * 0.7, 0.0)
        upper_1 = min(split + (1.0 - split) * 0.2, 1.0)
        upper_2 = min(split + (1.0 - split) * 0.45, 1.0)
        upper_3 = min(split + (1.0 - split) * 0.7, 1.0)
        colorscale = [
            [0.00, "#8c2d04"],
            [lower_1, "#cc4c02"],
            [lower_2, "#ec7014"],
            [split, "#fe9929"],
            [split, "#c7e9b4"],
            [upper_1, "#41b6c4"],
            [upper_2, "#1d91c0"],
            [upper_3, "#225ea8"],
            [1.00, "#081d58"],
        ]
        zmin = 0.0
        zmax = 1.0

    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=analysis_result["roi_order"],
            y=analysis_result["roi_order"],
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            text=value_text,
            customdata=pair_text,
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="%{customdata}<br>Value=%{z:.4f}<extra></extra>",
        )
    )

    if display_mode in {"p", "q"}:
        significance = np.array(analysis_result["significance_matrix"], dtype=float)
        xs: list[str] = []
        ys: list[str] = []
        for i, roi_from in enumerate(analysis_result["roi_order"]):
            for j, roi_to in enumerate(analysis_result["roi_order"]):
                if i == j or np.isnan(significance[i, j]) or significance[i, j] >= significance_threshold:
                    continue
                xs.append(roi_to)
                ys.append(roi_from)
        if xs:
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker={
                        "symbol": "square-open",
                        "size": 64,
                        "line": {"color": "#d62728", "width": 5},
                        "color": "rgba(0,0,0,0)",
                    },
                    hoverinfo="skip",
                    showlegend=False,
                    cliponaxis=False,
                )
            )

    figure.update_layout(
        title=f"{analysis_result['analysis_label']} | {analysis_result['freq_label']} | {display_label} | Selected: {selected_pair}",
        xaxis_title="ROI",
        yaxis_title="ROI",
        height=720,
        margin={"l": 80, "r": 20, "t": 60, "b": 60},
    )
    return figure


def _build_heatmap_description(
    analysis_result: dict[str, Any],
    display_mode: str,
    mtime_filter: str,
    significance_threshold: float,
) -> Any:
    group_text = (
        f"{analysis_result['selected_group_a']} vs {analysis_result['selected_group_b']}"
        if analysis_result["between_groups"]
        else str(analysis_result["selected_group_a"])
    )
    longitudinal_mode = "off"
    longitudinal_steps = ""
    if analysis_result["longitudinal_enabled"]:
        pairing_text = (
            "only subjects measured at both selected timepoints were retained"
            if analysis_result.get("longitudinal_require_pairs", True)
            else "all subjects available at the selected timepoints were retained, even if they were not measured twice"
        )
        longitudinal_mode = (
            f"on: {analysis_result['longitudinal_column']} {analysis_result['longitudinal_value_a']} vs "
            f"{analysis_result['longitudinal_value_b']} ({pairing_text})"
        )
        longitudinal_steps = (
            "4. Within each selected timepoint, compute the subject-level trial delta. "
            "5. Estimate the longitudinal contrast from those trial deltas using the selected repeated-subject or all-subject rule. "
        )

    if analysis_result["longitudinal_enabled"]:
        contrast_statement = (
            f"For each ROI pair and each subject-measurement, the within-measurement trial delta was first computed as "
            f"Trial {analysis_result['analysis_label'].split('Trial ')[1].split(' |')[0]}. "
            f"These subject-level trial deltas were then carried forward into the longitudinal comparison across "
            f"{analysis_result['longitudinal_column']} = {analysis_result['longitudinal_value_a']} and "
            f"{analysis_result['longitudinal_column']} = {analysis_result['longitudinal_value_b']} under the selected group setting {group_text}."
        )
        methods_statement = (
            f"Rows were restricted to the selected dataset, metric, frequency band ({analysis_result['freq_label']}), "
            f"and mtime filter ({mtime_filter}). For each subject, trial, and ROI pair, connectivity was averaged across all retained frequency bins. "
            f"For each available measurement, a subject-level trial delta was then computed as Trial B minus Trial A. "
            f"These subject-level trial deltas were compared across the two selected longitudinal states "
            f"({analysis_result['longitudinal_value_a']} vs {analysis_result['longitudinal_value_b']}). "
            f"When paired longitudinal analysis was enabled, only subjects observed at both selected timepoints were retained and the within-subject difference "
            f"between the two timepoints was estimated for each subject before inferential testing. "
            f"When paired longitudinal analysis was disabled, all available subjects at each selected timepoint were retained and the longitudinal contrast was estimated "
            f"from the timepoint-specific distributions of subject-level trial deltas. "
            f"If two groups were selected, the inferential target was the difference between groups in that longitudinal contrast. "
            f"For each ROI pair, inferential statistics were computed with {analysis_result['test_label']}, and multiplicity across ROI pairs was controlled with "
            f"{analysis_result['correction_label']}."
        )
        technical_steps = [
            f"1. Filter rows to dataset/metric, frequency band {analysis_result['freq_label']}, and mtime filter {mtime_filter}.",
            "2. Average connectivity across retained frequency bins for each subject, trial, and ROI pair.",
            "3. For each subject-measurement and ROI pair, compute the subject-level trial delta as Trial B minus Trial A.",
            (
                f"4. Construct the longitudinal contrast from these subject-level trial deltas for "
                f"{analysis_result['longitudinal_value_a']} vs {analysis_result['longitudinal_value_b']} under group setting {group_text}."
            ),
            (
                "5. If paired longitudinal mode is enabled, estimate a within-subject timepoint difference for each subject with both measurements; "
                "otherwise estimate the difference between the two timepoint-specific distributions of subject-level trial deltas."
            ),
            f"6. For each ROI pair, display {DISPLAY_OPTIONS[display_mode][0]} and compute inference with {analysis_result['test_label']}.",
            f"7. Adjust across ROI pairs with {analysis_result['correction_label']}.",
        ]
    elif analysis_result["between_groups"]:
        contrast_statement = (
            f"For each ROI pair and each subject, a subject-level contrast value was computed as Trial B minus Trial A. "
            f"The heatmap effect therefore represents the between-group difference in the mean of these subject-level contrast values "
            f"for {group_text}, not the raw difference between group-averaged connectivity matrices at a single trial."
        )
        methods_statement = (
            f"Rows were restricted to the selected dataset, metric, frequency band ({analysis_result['freq_label']}), and mtime filter ({mtime_filter}). "
            f"For each subject, trial, and ROI pair, connectivity was averaged across all retained frequency bins. "
            f"For each subject and ROI pair, a subject-level trial delta was then computed as Trial B minus Trial A. "
            f"This yielded one contrast value per subject and ROI pair. These subject-level contrast values were then separated by group "
            f"({group_text}), and for each ROI pair the inferential target was the difference between groups in the mean subject-level trial delta. "
            f"The statistical test applied at each ROI pair was {analysis_result['test_label']}. "
            f"The displayed effect size heatmap therefore summarizes the estimated between-group difference in mean subject-level contrast values, "
            f"whereas the p-value and q-value heatmaps summarize the corresponding inferential results. "
            f"Multiplicity across ROI pairs was controlled with {analysis_result['correction_label']}."
        )
        technical_steps = [
            f"1. Filter rows to dataset/metric, frequency band {analysis_result['freq_label']}, and mtime filter {mtime_filter}.",
            "2. Average connectivity across retained frequency bins for each subject, trial, and ROI pair.",
            "3. For each subject and ROI pair, compute the subject-level trial delta as Trial B minus Trial A.",
            f"4. Partition these subject-level trial deltas by group ({group_text}).",
            "5. For each ROI pair, estimate the between-group difference in the mean subject-level trial delta.",
            f"6. Apply {analysis_result['test_label']} to the subject-level trial deltas at each ROI pair.",
            f"7. Adjust across ROI pairs with {analysis_result['correction_label']}.",
        ]
    else:
        contrast_statement = (
            f"For each ROI pair and each subject, a subject-level contrast value was computed as Trial B minus Trial A. "
            f"The heatmap effect therefore represents the mean of these subject-level contrast values within {group_text}."
        )
        methods_statement = (
            f"Rows were restricted to the selected dataset, metric, frequency band ({analysis_result['freq_label']}), and mtime filter ({mtime_filter}). "
            f"For each subject, trial, and ROI pair, connectivity was averaged across all retained frequency bins. "
            f"For each subject and ROI pair, a subject-level trial delta was then computed as Trial B minus Trial A, yielding one contrast value per subject and ROI pair. "
            f"For each ROI pair, the effect estimate shown in the heatmap is the mean of these subject-level contrast values within the selected analysis set ({group_text}). "
            f"Inferential statistics at each ROI pair were computed with {analysis_result['test_label']}, and multiplicity across ROI pairs was controlled with "
            f"{analysis_result['correction_label']}."
        )
        technical_steps = [
            f"1. Filter rows to dataset/metric, frequency band {analysis_result['freq_label']}, and mtime filter {mtime_filter}.",
            "2. Average connectivity across retained frequency bins for each subject, trial, and ROI pair.",
            "3. For each subject and ROI pair, compute the subject-level trial delta as Trial B minus Trial A.",
            f"4. For each ROI pair, estimate the mean subject-level trial delta within {group_text}.",
            f"5. Apply {analysis_result['test_label']} to the subject-level trial deltas at each ROI pair.",
            f"6. Adjust across ROI pairs with {analysis_result['correction_label']}.",
        ]
    technical = " ".join(technical_steps)
    methods_narrative = (
        f"{contrast_statement} {methods_statement} "
        f"When longitudinal mode is active, it is {longitudinal_mode}. "
        f"The displayed matrix then shows either the estimated effect, the raw p-value, or the corrected q-value."
    )

    return html.Details(
        [
            html.Summary("Current heatmap calculation"),
            html.Div(
                [
                    html.Div("Short technical", style=_stats_section_title_style()),
                    html.P(technical),
                    html.Div("Methods-style description", style=_stats_section_title_style()),
                    html.P(methods_narrative),
                    html.P(
                        f"In p-value and q-value view, cells outlined in red are below {significance_threshold:.3f} under the currently selected correction rule."
                    ),
                ],
                style={"marginTop": "10px"},
            ),
        ],
        open=False,
    )


def _build_subject_bar_chart(pair_summary: dict[str, Any] | None, analysis_mode: str) -> go.Figure:
    figure = go.Figure()
    if not pair_summary:
        figure.update_layout(title="Subject values for selected edge")
        return figure

    detail_frame = pd.DataFrame(pair_summary["detail_records"])
    if detail_frame.empty:
        figure.update_layout(title="Subject values for selected edge")
        return figure

    if analysis_mode == "longitudinal_delta":
        id_column = "subject_id" if "subject_id" in detail_frame.columns else "base_subject_id"
        y_column = "delta"
        if {"measure_a_value", "measure_b_value"}.issubset(detail_frame.columns):
            chart_title = f"Longitudinal delta differences for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
            yaxis_title = "Delta(B) - Delta(A)"
        else:
            chart_title = f"Longitudinal trial deltas used for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
            yaxis_title = "Trial delta"
    else:
        id_column = "subject_id"
        y_column = "delta"
        chart_title = f"Subject deltas for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
        yaxis_title = "Trial B - Trial A"

    group_column = "plot_group_label" if "plot_group_label" in detail_frame.columns else "group_label"
    summary_group_column = "summary_group_label" if "summary_group_label" in detail_frame.columns else group_column
    if group_column not in detail_frame.columns:
        detail_frame[group_column] = "All"
    detail_frame[group_column] = detail_frame[group_column].astype(str)
    if summary_group_column not in detail_frame.columns:
        detail_frame[summary_group_column] = detail_frame[group_column]
    detail_frame[summary_group_column] = detail_frame[summary_group_column].astype(str)

    group_order = [entry["group_label"] for entry in pair_summary.get("group_stats", [])]
    if not group_order:
        group_order = list(dict.fromkeys(detail_frame[summary_group_column].tolist()))
    elif len(group_order) == 1 and not detail_frame[summary_group_column].isin(group_order).any():
        detail_frame[summary_group_column] = group_order[0]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    color_map = {group: palette[index % len(palette)] for index, group in enumerate(group_order)}

    category_array: list[str] = []
    has_multiple_groups = len(group_order) > 1
    for index, group_name in enumerate(group_order):
        group_frame = detail_frame[detail_frame[summary_group_column] == group_name].copy()
        if group_frame.empty:
            continue
        group_frame[id_column] = group_frame[id_column].astype(str)
        figure.add_trace(
            go.Bar(
                x=group_frame[id_column],
                y=group_frame[y_column],
                name=group_name,
                marker={"color": color_map[group_name]},
                customdata=np.column_stack(
                    [
                        group_frame[id_column],
                        group_frame[group_column],
                    ]
                ),
                hovertemplate="ID=%{customdata[0]}<br>Group=%{customdata[1]}<br>Delta=%{y:.4f}<extra></extra>",
            )
        )
        category_array.extend(group_frame[id_column].tolist())
        if has_multiple_groups and index < len(group_order) - 1:
            category_array.append(f"__gap__{index}")
            figure.add_trace(
                go.Bar(
                    x=[f"__gap__{index}"],
                    y=[0],
                    name="",
                    marker={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    tick_text = ["" if value.startswith("__gap__") else value for value in category_array]
    figure.update_layout(
        title=chart_title,
        xaxis={
            "title": "Subject",
            "categoryorder": "array",
            "categoryarray": category_array,
            "tickmode": "array",
            "tickvals": category_array,
            "ticktext": tick_text,
        },
        yaxis_title=yaxis_title,
        barmode="overlay",
        bargap=0.18,
        height=420,
        legend_title="Group",
    )
    return figure


def _build_subject_bar_stats(pair_summary: dict[str, Any] | None) -> Any:
    if not pair_summary:
        return html.P("Select an edge in the heatmap to view descriptive and inferential statistics.")

    def format_stat(value: Any, digits: int = 6) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    group_stats = pair_summary.get("group_stats", [])
    if not group_stats:
        group_stats = [
            {
                "group_label": "Group 1",
                "n": pair_summary.get("n_group_a", pair_summary["n"]),
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "sem": np.nan,
            }
        ]

    metric_rows = [
        ("n", [str(group["n"]) for group in group_stats]),
        ("Mean", [format_stat(group["mean"]) for group in group_stats]),
        ("Median", [format_stat(group["median"]) for group in group_stats]),
        ("SD", [format_stat(group["std"]) for group in group_stats]),
        ("SEM", [format_stat(group["sem"]) for group in group_stats]),
    ]

    inferential_rows = [
        ("Total n", str(pair_summary["n"])),
        ("t", format_stat(pair_summary.get("statistic"))),
        ("Effect size", format_stat(pair_summary.get("effect_size"))),
        ("p uncorr.", format_stat(pair_summary.get("p_value"))),
        ("q FDR-BH", format_stat(pair_summary.get("q_value_fdr_bh"))),
        ("q Bonferroni", format_stat(pair_summary.get("q_value_bonferroni"))),
        ("q Holm", format_stat(pair_summary.get("q_value_holm"))),
    ]

    descriptive_header = [
        html.Thead(
            html.Tr(
                [html.Th("Metric", style=_stats_cell_style(header=True))]
                + [html.Th(group["group_label"], style=_stats_cell_style(header=True)) for group in group_stats]
            )
        )
    ]
    descriptive_body = [
        html.Tbody(
            [
                html.Tr(
                    [html.Td(label, style=_stats_cell_style(label=True))]
                    + [html.Td(value, style=_stats_cell_style()) for value in values]
                )
                for label, values in metric_rows
            ]
        )
    ]

    inferential_table = html.Table(
        [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(label, style=_stats_cell_style(label=True)),
                            html.Td(value, style=_stats_cell_style()),
                        ]
                    )
                    for label, value in inferential_rows
                ]
            )
        ],
        style=_stats_table_style(),
    )

    return html.Div(
        [
            html.H4(f"Statistics for {pair_summary['roi_from']} -> {pair_summary['roi_to']}", style={"marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Descriptive", style=_stats_section_title_style()),
                            html.Table(descriptive_header + descriptive_body, style=_stats_table_style()),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("Inferential", style=_stats_section_title_style()),
                            inferential_table,
                        ]
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "minmax(0, 2fr) minmax(240px, 1fr)", "gap": "14px"},
            ),
        ]
    )


def _build_heatmap_correlation_view(
    covariate_result: dict[str, Any],
    correlation_variable: str | None,
) -> tuple[go.Figure, Any]:
    figure = go.Figure()
    if covariate_result.get("message"):
        figure.update_layout(title="Delta vs variable", height=420)
        return figure, html.P(covariate_result["message"])

    correlation = covariate_result.get("correlation")
    if not correlation_variable:
        figure.update_layout(title="Delta vs variable", height=420)
        return figure, html.P("Select a variable to compute correlation statistics for the selected edge.")
    if not correlation:
        figure.update_layout(title=f"Delta vs {correlation_variable}", height=420)
        return figure, html.P("No correlation output available.")
    if "message" in correlation:
        figure.update_layout(title=f"Delta vs {correlation_variable}", height=420)
        return figure, html.P(correlation["message"])
    if "x" not in correlation or "y" not in correlation:
        figure.update_layout(title=f"Delta vs {correlation_variable}", height=420)
        return figure, html.P("No scatter data available.")

    x = np.asarray(correlation["x"], dtype=float)
    y = np.asarray(correlation["y"], dtype=float)
    point_labels = correlation.get("point_labels", [])
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    point_labels = [label for label, keep in zip(point_labels, mask.tolist(), strict=False) if keep] if point_labels else []
    if len(x) < 3:
        figure.update_layout(title=f"Delta vs {correlation_variable}", height=420)
        return figure, html.P("Not enough complete observations for correlation.")

    group_results = covariate_result.get("correlation_by_group", [])
    has_group_fits = any("message" not in group and "x" in group and "y" in group for group in group_results)
    if has_group_fits:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        for index, group in enumerate(group_results):
            if "message" in group or "x" not in group or "y" not in group:
                continue
            group_x = np.asarray(group["x"], dtype=float)
            group_y = np.asarray(group["y"], dtype=float)
            group_order = np.argsort(group_x)
            color = palette[index % len(palette)]
            figure.add_trace(
                go.Scatter(
                    x=group_x.tolist(),
                    y=group_y.tolist(),
                    mode="markers",
                    name=f"{group['group_label']} subjects",
                    text=group.get("point_labels"),
                    hovertemplate="Point=%{text}<br>X=%{x:.4f}<br>Delta=%{y:.4f}<extra></extra>",
                    marker={"size": 8, "color": color, "opacity": 0.85},
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=group_x[group_order].tolist(),
                    y=(group["intercept"] + group["slope"] * group_x[group_order]).tolist(),
                    mode="lines",
                    name=f"{group['group_label']} fit",
                    line={"color": color, "width": 2},
                )
            )
        overall_order = np.argsort(x)
        figure.add_trace(
            go.Scatter(
                x=x[overall_order].tolist(),
                y=(correlation["intercept"] + correlation["slope"] * x[overall_order]).tolist(),
                mode="lines",
                name="All subjects fit",
                line={"color": "#444444", "width": 2, "dash": "dash"},
            )
        )
    else:
        order = np.argsort(x)
        x_sorted = x[order]
        y_hat_sorted = correlation["intercept"] + correlation["slope"] * x_sorted
        figure.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=y.tolist(),
                mode="markers",
                name="Subjects",
                text=point_labels,
                hovertemplate="Point=%{text}<br>X=%{x:.4f}<br>Delta=%{y:.4f}<extra></extra>",
                marker={"size": 8, "color": "#1f77b4", "opacity": 0.85},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_sorted.tolist(),
                y=y_hat_sorted.tolist(),
                mode="lines",
                name="Linear fit",
                line={"color": "#d62728", "width": 2},
            )
        )
    figure.update_layout(
        title=f"Connectivity delta vs {correlation_variable}",
        xaxis_title=correlation_variable,
        yaxis_title="Connectivity delta",
        height=420,
        legend={"orientation": "h", "x": 0, "y": 1.12},
    )

    stats_sections = [
        html.Div(
            [
                html.Div("All subjects", style=_stats_section_title_style()),
                _build_correlation_stats_table(correlation),
            ]
        )
    ]

    valid_group_sections = []
    for group in group_results:
        valid_group_sections.append(
            html.Div(
                [
                    html.Div(str(group.get("group_label", "Group")), style=_stats_section_title_style()),
                    _build_correlation_stats_table(group),
                ]
            )
        )
    if valid_group_sections:
        stats_sections.extend(valid_group_sections)

    comparison = covariate_result.get("correlation_group_comparison")
    if comparison:
        stats_sections.append(
            html.Div(
                [
                    html.Div("Group comparison", style=_stats_section_title_style()),
                    _build_group_correlation_comparison_table(comparison),
                ]
            )
        )

    return figure, html.Div(
        [
            html.H4("Correlation statistics"),
            html.Div(
                stats_sections,
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))", "gap": "14px"},
            ),
        ]
    )


def _build_heatmap_regression_view(regression_result: dict[str, Any]) -> tuple[Any, go.Figure, go.Figure]:
    fit_figure = go.Figure()
    residual_figure = go.Figure()

    if regression_result.get("message"):
        fit_figure.update_layout(title="Observed vs fitted", height=360)
        residual_figure.update_layout(title="Residuals vs fitted", height=360)
        return html.P(regression_result["message"]), fit_figure, residual_figure

    summary_lines = [
        html.H4("Regression summary"),
        html.P(regression_result["model_label"]),
        html.P(f"Formula: {regression_result['formula']}"),
        html.P(f"Response: {regression_result['response_definition']}"),
        html.P(f"Observation unit: {regression_result['observation_unit']}"),
        html.P(regression_result["methods_description"]),
        html.P(f"n = {regression_result['n']}, R² = {regression_result['r_squared']:.6f}, adjusted R² = {regression_result['adj_r_squared']:.6f}"),
    ]
    if regression_result.get("primary_effect"):
        primary = regression_result["primary_effect"]
        summary_lines.append(
            html.P(
                f"Primary design effect: {primary['label']} | estimate = {primary['estimate']:.6f}, "
                f"t = {primary['t_value']:.6f}, p = {primary['p_value']:.6g}"
            )
        )
    elif regression_result.get("design_effect_interpretation"):
        summary_lines.append(html.P(regression_result["design_effect_interpretation"]))

    coefficient_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Term", style=_stats_cell_style(header=True)),
                        html.Th("Estimate", style=_stats_cell_style(header=True)),
                        html.Th("SE", style=_stats_cell_style(header=True)),
                        html.Th("t", style=_stats_cell_style(header=True)),
                        html.Th("p", style=_stats_cell_style(header=True)),
                        html.Th("95% CI", style=_stats_cell_style(header=True)),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row["label"], style=_stats_cell_style(label=True)),
                            html.Td(f"{row['estimate']:.6f}", style=_stats_cell_style()),
                            html.Td(f"{row['std_error']:.6f}", style=_stats_cell_style()),
                            html.Td(f"{row['t_value']:.6f}", style=_stats_cell_style()),
                            html.Td(f"{row['p_value']:.6g}", style=_stats_cell_style()),
                            html.Td(f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]", style=_stats_cell_style()),
                        ]
                    )
                    for row in regression_result["coefficients"]
                ]
            ),
        ],
        style=_stats_table_style(),
    )

    diagnostics = regression_result["diagnostics"]
    observed = diagnostics["observed"]
    fitted = diagnostics["fitted"]
    residuals = diagnostics["residuals"]
    point_labels = diagnostics.get("point_labels")

    fit_figure.add_trace(
        go.Scatter(
            x=fitted,
            y=observed,
            mode="markers",
            text=point_labels,
            marker={"size": 8, "color": "#1f77b4", "opacity": 0.85},
            hovertemplate="Point=%{text}<br>Fitted=%{x:.4f}<br>Observed=%{y:.4f}<extra></extra>",
            name="Observations",
        )
    )
    if fitted:
        min_axis = min(min(fitted), min(observed))
        max_axis = max(max(fitted), max(observed))
        fit_figure.add_trace(
            go.Scatter(
                x=[min_axis, max_axis],
                y=[min_axis, max_axis],
                mode="lines",
                line={"color": "#444444", "dash": "dash"},
                name="Identity",
                hoverinfo="skip",
            )
        )
    fit_figure.update_layout(
        title="Observed vs fitted",
        xaxis_title="Fitted value",
        yaxis_title="Observed value",
        height=360,
    )

    residual_figure.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            text=point_labels,
            marker={"size": 8, "color": "#ff7f0e", "opacity": 0.85},
            hovertemplate="Point=%{text}<br>Fitted=%{x:.4f}<br>Residual=%{y:.4f}<extra></extra>",
            name="Residuals",
        )
    )
    if fitted:
        residual_figure.add_trace(
            go.Scatter(
                x=[min(fitted), max(fitted)],
                y=[0, 0],
                mode="lines",
                line={"color": "#444444", "dash": "dash"},
                name="Zero",
                hoverinfo="skip",
            )
        )
    residual_figure.update_layout(
        title="Residuals vs fitted",
        xaxis_title="Fitted value",
        yaxis_title="Residual",
        height=360,
    )

    if diagnostics.get("message"):
        summary_lines.append(html.P(diagnostics["message"]))

    return html.Div(summary_lines + [coefficient_table]), fit_figure, residual_figure


def _stats_table_style() -> dict[str, str]:
    return {
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "13px",
        "backgroundColor": "#ffffff",
        "border": "1px solid #d9dde3",
    }


def _build_correlation_stats_table(correlation: dict[str, Any]) -> html.Table:
    if "message" in correlation:
        rows = [("Status", correlation["message"]), ("n", str(correlation.get("n", "n/a")))]
    else:
        rows = [
            ("Method", f"{correlation['method'].title()} correlation"),
            ("n", str(correlation["n"])),
            ("r / rho", f"{correlation['statistic']:.6f}"),
            ("p-value", f"{correlation['p_value']:.6g}"),
            ("Slope", f"{correlation['slope']:.6f}"),
            ("Intercept", f"{correlation['intercept']:.6f}"),
            ("R^2", f"{correlation['r_squared']:.6f}"),
            ("Slope stderr", f"{correlation['slope_stderr']:.6f}" if not pd.isna(correlation["slope_stderr"]) else "n/a"),
            ("Intercept stderr", f"{correlation['intercept_stderr']:.6f}" if not pd.isna(correlation["intercept_stderr"]) else "n/a"),
            ("Fit p-value", f"{correlation['fit_p_value']:.6g}"),
            ("RMSE", f"{correlation['rmse']:.6f}"),
        ]

    return html.Table(
        [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(label, style=_stats_cell_style(label=True)),
                            html.Td(value, style=_stats_cell_style()),
                        ]
                    )
                    for label, value in rows
                ]
            )
        ],
        style=_stats_table_style(),
    )


def _build_group_correlation_comparison_table(comparison: dict[str, Any]) -> html.Table:
    if "message" in comparison:
        rows = [("Status", comparison["message"]), ("Method", comparison.get("method", "n/a").title())]
    else:
        rows = [
            ("Test", comparison["test"]),
            ("Method", comparison["method"].title()),
            (comparison["group_a"], f"r / rho = {comparison['statistic_a']:.6f}, n = {comparison['n_group_a']}"),
            (comparison["group_b"], f"r / rho = {comparison['statistic_b']:.6f}, n = {comparison['n_group_b']}"),
            ("z", f"{comparison['z_statistic']:.6f}"),
            ("p-value", f"{comparison['p_value']:.6g}"),
        ]

    return html.Table(
        [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(label, style=_stats_cell_style(label=True)),
                            html.Td(value, style=_stats_cell_style()),
                        ]
                    )
                    for label, value in rows
                ]
            )
        ],
        style=_stats_table_style(),
    )


def _stats_cell_style(*, header: bool = False, label: bool = False) -> dict[str, str]:
    style = {
        "padding": "7px 10px",
        "border": "1px solid #d9dde3",
        "textAlign": "left",
        "verticalAlign": "middle",
    }
    if header:
        style.update({"backgroundColor": "#eef2f7", "fontWeight": "600"})
    elif label:
        style.update({"backgroundColor": "#f8fafc", "fontWeight": "600", "whiteSpace": "nowrap"})
    return style


def _stats_section_title_style() -> dict[str, str]:
    return {
        "fontSize": "12px",
        "fontWeight": "700",
        "letterSpacing": "0.04em",
        "textTransform": "uppercase",
        "color": "#445066",
        "marginBottom": "6px",
    }


def _build_detail_view(pair_summary: dict[str, Any] | None, analysis_mode: str) -> tuple[Any, go.Figure]:
    if not pair_summary:
        return html.P("Select an edge in the heatmap to inspect details."), go.Figure()

    summary = html.Div(
        [
            html.H4(f"{pair_summary['roi_from']} -> {pair_summary['roi_to']}"),
            html.P(f"n = {pair_summary['n']}"),
            html.P(f"n(group 1) = {pair_summary.get('n_group_a', pair_summary['n'])}, n(group 2) = {pair_summary.get('n_group_b', pair_summary['n'])}"),
            html.P(f"Mean delta = {pair_summary['mean_delta']:.6f}"),
            html.P(f"Effect size = {pair_summary['effect_size']:.6f}" if not np.isnan(pair_summary["effect_size"]) else "Effect size = n/a"),
            html.P(f"t = {pair_summary['statistic']:.6f}" if not np.isnan(pair_summary["statistic"]) else "t = n/a"),
            html.P(f"p = {pair_summary['p_value']:.6g}" if not np.isnan(pair_summary["p_value"]) else "p = n/a"),
            html.P(f"q = {pair_summary['q_value']:.6g}" if not np.isnan(pair_summary["q_value"]) else "q = n/a"),
        ]
    )

    detail_frame = pd.DataFrame(pair_summary["detail_records"])
    figure = go.Figure()
    if not detail_frame.empty:
        if analysis_mode != "longitudinal_delta":
            figure.add_trace(
                go.Scatter(
                    x=detail_frame["trial_a_value"],
                    y=detail_frame["trial_b_value"],
                    mode="markers+text",
                    text=detail_frame["subject_id"],
                    textposition="top center",
                    name="Subjects",
                )
            )
            figure.update_layout(
                title="Trial A vs Trial B",
                xaxis_title="Trial A value",
                yaxis_title="Trial B value",
                height=420,
            )
        elif {"measure_a_value", "measure_b_value"}.issubset(detail_frame.columns):
            figure.add_trace(
                go.Scatter(
                    x=detail_frame["measure_a_value"],
                    y=detail_frame["measure_b_value"],
                    mode="markers+text",
                    text=detail_frame["base_subject_id"],
                    textposition="top center",
                    name="Subjects",
                )
            )
            figure.update_layout(
                title="Longitudinal delta A vs B",
                xaxis_title="Delta at value A",
                yaxis_title="Delta at value B",
                height=420,
            )
        else:
            group_column = "plot_group_label" if "plot_group_label" in detail_frame.columns else "group_label"
            id_column = "subject_id" if "subject_id" in detail_frame.columns else "base_subject_id"
            figure.add_trace(
                go.Scatter(
                    x=detail_frame[group_column],
                    y=detail_frame["delta"],
                    mode="markers+text",
                    text=detail_frame[id_column],
                    textposition="top center",
                    name="Observations",
                )
            )
            figure.update_layout(
                title="Longitudinal observations used in the estimate",
                xaxis_title="Subgroup",
                yaxis_title="Trial delta",
                height=420,
            )
    return summary, figure


def _build_covariate_view(
    covariate_result: dict[str, Any],
    analysis_mode: str,
    correlation_variable: str | None,
) -> tuple[Any, go.Figure]:
    if covariate_result.get("message"):
        return html.P(covariate_result["message"]), go.Figure()

    lines = [html.H4("Covariate analysis")]
    if analysis_mode == "longitudinal_delta":
        lines.append(html.P("Session contrasts use baseline covariates from the earliest available measurement."))

    if "correlation" in covariate_result:
        correlation = covariate_result["correlation"]
        if "message" in correlation:
            lines.append(html.P(correlation["message"]))
        else:
            lines.extend(
                [
                    html.P(f"{correlation['method'].title()} correlation with {correlation['variable']}"),
                    html.P(f"n = {correlation['n']}"),
                    html.P(f"r = {correlation['statistic']:.6f}, p = {correlation['p_value']:.6g}"),
                ]
            )

    if "regression" in covariate_result:
        regression = covariate_result["regression"]
        if "message" in regression:
            lines.append(html.P(regression["message"]))
        else:
            lines.append(html.P(f"OLS regression on {', '.join(regression['variables'])}"))
            lines.append(html.P(f"n = {regression['n']}, R² = {regression['r_squared']:.6f}"))
            for name, values in regression["coefficients"].items():
                lines.append(html.P(f"{name}: coef = {values['coef']:.6f}, p = {values['p_value']:.6g}"))

    figure = go.Figure()
    correlation = covariate_result.get("correlation")
    if correlation and "x" in correlation:
        figure.add_trace(
            go.Scatter(
                x=correlation["x"],
                y=correlation["y"],
                mode="markers",
                name="Subjects",
                text=correlation.get("point_labels"),
                hovertemplate="Point=%{text}<br>X=%{x:.4f}<br>Delta=%{y:.4f}<extra></extra>",
            )
        )
        figure.update_layout(
            title=f"Delta vs {correlation_variable}",
            xaxis_title=correlation_variable,
            yaxis_title="Connectivity delta",
            height=420,
        )

    return html.Div(lines), figure
