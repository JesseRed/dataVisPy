"""Dash application factory for the connectivity analysis dashboard."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from dash import Dash, Input, Output, State, callback_context, dcc, html

from data_vis_py.io.dataset_loader import DatasetBundle, list_dataset_files, load_dataset
from data_vis_py.stats.analysis import (
    ALL_GROUPS_LABEL,
    ALL_GROUPS_VALUE,
    AnalysisConfig,
    LongitudinalDetailConfig,
    NetworkAnalysisConfig,
    PatternsAnalysisConfig,
    SELECTED_EDGE_DELTA_LABEL,
    SELECTED_EDGE_DELTA_VALUE,
    run_analysis,
    run_covariate_analysis,
    run_leave_one_out_analysis,
    run_longitudinal_detail_analysis,
    run_multivariate_regression_analysis,
    run_network_analysis,
    run_patterns_analysis,
    run_regression_influence_analysis,
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

NETWORK_MODE_OPTIONS = [
    {"label": "Summary", "value": "summary"},
    {"label": "Graph", "value": "graph"},
    {"label": "Modules", "value": "modules"},
    {"label": "NBS", "value": "nbs"},
]

NETWORK_METRIC_OPTIONS = [
    {"label": "ROI mean connectivity", "value": "roi_mean_connectivity"},
    {"label": "Within-network connectivity", "value": "within_network_connectivity"},
    {"label": "Between-network connectivity", "value": "between_network_connectivity"},
    {"label": "Laterality", "value": "laterality_index"},
    {"label": "Node strength", "value": "node_strength"},
    {"label": "Degree", "value": "degree"},
    {"label": "Betweenness", "value": "betweenness"},
    {"label": "Clustering coefficient", "value": "clustering_coefficient"},
    {"label": "Local efficiency", "value": "local_efficiency"},
    {"label": "Global efficiency", "value": "global_efficiency"},
    {"label": "Participation", "value": "participation"},
    {"label": "Hubness", "value": "hubness"},
]

PATTERNS_MODE_OPTIONS = [
    {"label": "Embedding", "value": "embedding"},
    {"label": "Subject clusters", "value": "subject_clusters"},
    {"label": "Feature patterns", "value": "feature_patterns"},
    {"label": "CCA / PLS", "value": "brain_behavior"},
]


def create_dashboard(
    raw_root: Path,
    dataset_ids: list[str],
    initial_dataset: str,
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
            dcc.Store(id="excluded-idx-store", data=[]),
            dcc.Store(id="outlier-top3-store", data=[]),
            html.Div(
                [
                    html.H2("MEG Connectivity Explorer"),
                    html.P("Interactive paired contrasts with edge, network, and covariate drill-down analysis for MEG connectivity datasets."),
                    _sidebar(initial_bundle, dataset_ids, initial_dataset, numeric_subject_columns, group_options, json_files, csv_files, initial_json, initial_csv),
                    html.Div(id="analysis-summary", className="panel"),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="main-tabs",
                        value="heatmap",
                        children=[
                            dcc.Tab(
                                label="Heatmap",
                                value="heatmap",
                                children=[
                                    html.Div(id="heatmap-status", className="panel"),
                                    dcc.Loading(
                                        children=html.Div(
                                            [
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
                                                html.Details(
                                                    [
                                                        html.Summary("Longitudinal modeling"),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label("Longitudinal model"),
                                                                        dcc.Dropdown(
                                                                            id="heatmap-longitudinal-model",
                                                                            options=[
                                                                                {"label": "Mixed effects", "value": "mixed_effects"},
                                                                                {"label": "Change score", "value": "change_score"},
                                                                                {"label": "ANCOVA", "value": "ancova"},
                                                                            ],
                                                                            value="mixed_effects",
                                                                            clearable=False,
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="heatmap-longitudinal-random-slope",
                                                                            options=[{"label": "Request random slope for time", "value": "enabled"}],
                                                                            value=[],
                                                                            style={"marginTop": "10px"},
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="heatmap-longitudinal-show-trajectories",
                                                                            options=[{"label": "Show trajectories", "value": "enabled"}],
                                                                            value=["enabled"],
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="heatmap-longitudinal-show-reliable-change",
                                                                            options=[{"label": "Show reliable change", "value": "enabled"}],
                                                                            value=["enabled"],
                                                                        ),
                                                                    ],
                                                                    className="panel",
                                                                ),
                                                                html.Div(id="heatmap-longitudinal-summary", className="panel"),
                                                                dcc.Graph(id="heatmap-longitudinal-fit-chart"),
                                                                dcc.Graph(id="heatmap-longitudinal-residual-chart"),
                                                                html.Div(id="heatmap-longitudinal-robustness-summary", className="panel"),
                                                                html.Div(id="heatmap-longitudinal-trajectory-summary", className="panel"),
                                                                dcc.Graph(id="heatmap-longitudinal-trajectory-chart"),
                                                                html.Div(id="heatmap-reliable-change-summary", className="panel"),
                                                                dcc.Graph(id="heatmap-reliable-change-chart"),
                                                            ],
                                                            style={"marginTop": "10px"},
                                                        ),
                                                    ],
                                                    open=False,
                                                    className="panel",
                                                ),
                                            ]
                                        ),
                                        type="default",
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Outlier",
                                value="outlier",
                                children=[
                                    html.Div(id="outlier-status", className="panel"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Outlier view"),
                                                    dcc.RadioItems(
                                                        id="outlier-mode",
                                                        options=[
                                                            {"label": "Global influence", "value": "global"},
                                                            {"label": "Selected ROI pair", "value": "pair"},
                                                            {"label": "Regression diagnostics", "value": "regression"},
                                                        ],
                                                        value="global",
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Exclude persons globally (IDX)"),
                                                            dcc.Checklist(id="excluded-idx-checklist", options=[], value=[]),
                                                        ],
                                                        style={"maxHeight": "260px", "overflowY": "auto"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button("Reset exclusions", id="outlier-reset-exclusions", n_clicks=0),
                                                            html.Button("Mark top 3 influential", id="outlier-top3-button", n_clicks=0, style={"marginLeft": "10px"}),
                                                        ],
                                                        style={"marginTop": "12px"},
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        style={"display": "grid", "gridTemplateColumns": "minmax(280px, 360px) minmax(0, 1fr)", "gap": "14px"},
                                    ),
                                    dcc.Loading(
                                        children=html.Div(
                                            [
                                                html.Div(id="outlier-overview", className="panel"),
                                                dcc.Graph(id="outlier-global-chart"),
                                                html.Div(id="outlier-pair-summary", className="panel"),
                                                dcc.Graph(id="outlier-pair-chart"),
                                                html.Div(id="outlier-regression-summary", className="panel"),
                                                dcc.Graph(id="outlier-regression-leverage-chart"),
                                                dcc.Graph(id="outlier-regression-cook-chart"),
                                            ]
                                        ),
                                        type="default",
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Network",
                                value="network",
                                children=[
                                    html.Div(id="network-status", className="panel"),
                                    html.Details(
                                        [
                                            html.Summary("Help / Explanation"),
                                            html.Div(_build_network_help_content(), style={"marginTop": "10px"}),
                                        ],
                                        open=False,
                                        className="panel",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Network analysis mode"),
                                                    dcc.RadioItems(id="network-mode", options=NETWORK_MODE_OPTIONS, value="summary"),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Score family / metric"),
                                                    dcc.Dropdown(id="network-metric-name", options=NETWORK_METRIC_OPTIONS, value="roi_mean_connectivity", clearable=False),
                                                    html.Label("Weight mode"),
                                                    dcc.Dropdown(
                                                        id="network-weight-mode",
                                                        options=[
                                                            {"label": "Positive weights", "value": "positive"},
                                                            {"label": "Absolute weights", "value": "absolute"},
                                                            {"label": "Raw positive part", "value": "raw"},
                                                        ],
                                                        value="positive",
                                                        clearable=False,
                                                    ),
                                                    html.Label("Threshold mode"),
                                                    dcc.Dropdown(
                                                        id="network-threshold-mode",
                                                        options=[
                                                            {"label": "None", "value": "none"},
                                                            {"label": "Absolute weight", "value": "absolute weight"},
                                                            {"label": "Density", "value": "density"},
                                                        ],
                                                        value="none",
                                                        clearable=False,
                                                    ),
                                                    html.Label("Threshold value"),
                                                    dcc.Input(id="network-threshold-value", type="number", value=0.2, step=0.01, min=0.0, max=1.0),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("NBS primary threshold"),
                                                    dcc.Input(id="network-nbs-primary-threshold", type="number", value=2.0, step=0.1, min=0.0),
                                                    html.Label("NBS permutations"),
                                                    dcc.Input(id="network-nbs-permutations", type="number", value=200, step=10, min=10),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(220px, 1fr))", "gap": "14px"},
                                    ),
                                    dcc.Loading(
                                        children=html.Div(
                                            [
                                                html.Div(id="network-overview", className="panel"),
                                                html.Div(id="network-primary-summary", className="panel"),
                                                dcc.Graph(id="network-primary-chart"),
                                                html.Div(id="network-secondary-summary", className="panel"),
                                                dcc.Graph(id="network-secondary-chart"),
                                            ]
                                        ),
                                        type="default",
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Patterns",
                                value="patterns",
                                children=[
                                    html.Div(id="patterns-status", className="panel"),
                                    html.Details(
                                        [
                                            html.Summary("Help / Explanation"),
                                            html.Div(_build_patterns_help_content(), style={"marginTop": "10px"}),
                                        ],
                                        open=False,
                                        className="panel",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Pattern mode"),
                                                    dcc.RadioItems(id="patterns-mode", options=PATTERNS_MODE_OPTIONS, value="embedding"),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Embedding method"),
                                                    dcc.Dropdown(
                                                        id="patterns-embedding-method",
                                                        options=[
                                                            {"label": "PCA", "value": "pca"},
                                                            {"label": "t-SNE", "value": "tsne"},
                                                            {"label": "UMAP", "value": "umap"},
                                                        ],
                                                        value="pca",
                                                        clearable=False,
                                                    ),
                                                    html.Label("Embedding dimension"),
                                                    dcc.Dropdown(
                                                        id="patterns-embedding-dim",
                                                        options=[{"label": "2D", "value": 2}, {"label": "3D", "value": 3}],
                                                        value=2,
                                                        clearable=False,
                                                    ),
                                                    html.Label("Color variable"),
                                                    dcc.Dropdown(id="patterns-color-variable", options=[], value="group_label", clearable=False),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Cluster method"),
                                                    dcc.Dropdown(
                                                        id="patterns-cluster-method",
                                                        options=[
                                                            {"label": "k-means", "value": "kmeans"},
                                                            {"label": "Hierarchical", "value": "hierarchical"},
                                                            {"label": "Gaussian mixture", "value": "gaussian_mixture"},
                                                        ],
                                                        value="kmeans",
                                                        clearable=False,
                                                    ),
                                                    html.Label("Number of clusters"),
                                                    dcc.Input(id="patterns-n-clusters", type="number", value=3, step=1, min=2),
                                                    html.Label("Feature pattern level"),
                                                    dcc.Dropdown(
                                                        id="patterns-feature-level",
                                                        options=[
                                                            {"label": "Edges", "value": "edges"},
                                                            {"label": "ROIs", "value": "rois"},
                                                        ],
                                                        value="edges",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Multivariate method"),
                                                    dcc.Dropdown(
                                                        id="patterns-brain-behavior-method",
                                                        options=[{"label": "PLS", "value": "pls"}, {"label": "CCA", "value": "cca"}],
                                                        value="pls",
                                                        clearable=False,
                                                    ),
                                                    html.Label("Behavior variables"),
                                                    dcc.Dropdown(id="patterns-behavior-variables", options=[], value=[], multi=True),
                                                    html.Label("Components"),
                                                    dcc.Input(id="patterns-n-components", type="number", value=2, step=1, min=1),
                                                    html.Label("Standardization"),
                                                    dcc.Dropdown(
                                                        id="patterns-standardize",
                                                        options=[
                                                            {"label": "None", "value": "none"},
                                                            {"label": "Z-score by feature", "value": "zscore"},
                                                        ],
                                                        value="zscore",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(260px, 1fr))", "gap": "14px"},
                                    ),
                                    dcc.Loading(
                                        children=html.Div(
                                            [
                                                html.Div(id="patterns-overview", className="panel"),
                                                html.Div(id="patterns-primary-summary", className="panel"),
                                                dcc.Graph(id="patterns-primary-chart"),
                                                html.Div(id="patterns-secondary-summary", className="panel"),
                                                dcc.Graph(id="patterns-secondary-chart"),
                                            ]
                                        ),
                                        type="default",
                                    ),
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

    def current_dataset_dir(dataset_id: str) -> Path:
        return raw_root / dataset_id

    def current_bundle(dataset_id: str, json_filename: str, csv_filename: str) -> DatasetBundle:
        return load_dataset(current_dataset_dir(dataset_id), json_filename=json_filename, csv_filename=csv_filename)

    def resolve_analysis_context(
        *,
        dataset_id: str,
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
        excluded_idx: list[str] | None,
    ) -> tuple[DatasetBundle, AnalysisConfig, dict[str, Any], str]:
        bundle = current_bundle(dataset_id, json_filename, csv_filename)
        longitudinal_enabled = "enabled" in (longitudinal_enabled_flags or [])
        longitudinal_require_pairs = "paired_only" in (longitudinal_pairing_flags or [])
        if BAND_PRESETS.get(band_preset):
            freq_min, freq_max = BAND_PRESETS[band_preset]
        positive_frequencies = [frequency for frequency in bundle.frequencies if frequency > 0]
        min_frequency = min(positive_frequencies) if positive_frequencies else min(bundle.frequencies)
        max_frequency = max(positive_frequencies) if positive_frequencies else max(bundle.frequencies)
        freq_min = float(freq_min if freq_min is not None else min_frequency)
        freq_max = float(freq_max if freq_max is not None else max_frequency)
        freq_min = min(max(freq_min, min_frequency), max_frequency)
        freq_max = min(max(freq_max, min_frequency), max_frequency)
        if freq_min > freq_max:
            freq_min, freq_max = freq_max, freq_min
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
            excluded_idx=tuple(sorted({str(value) for value in (excluded_idx or [])})),
        )
        analysis_result = run_analysis(bundle, config)
        effective_analysis_mode = "longitudinal_delta" if longitudinal_enabled else "trial_delta"
        return bundle, config, analysis_result, effective_analysis_mode

    @app.callback(
        Output("json-file", "options"),
        Output("json-file", "value"),
        Output("csv-file", "options"),
        Output("csv-file", "value"),
        Input("dataset-id", "value"),
        State("json-file", "value"),
        State("csv-file", "value"),
    )
    def sync_dataset_files(dataset_id: str, current_json: str | None, current_csv: str | None):
        files = list_dataset_files(current_dataset_dir(dataset_id))
        json_value = current_json if current_json in files["json_files"] else ("data_coh.json" if "data_coh.json" in files["json_files"] else files["json_files"][0])
        csv_value = current_csv if current_csv in files["csv_files"] else ("info.csv" if "info.csv" in files["csv_files"] else files["csv_files"][0])
        return (
            [{"label": name, "value": name} for name in files["json_files"]],
            json_value,
            [{"label": name, "value": name} for name in files["csv_files"]],
            csv_value,
        )

    @app.callback(
        Output("metric-name", "options"),
        Output("metric-name", "value"),
        Output("trial-a", "options"),
        Output("trial-a", "value"),
        Output("trial-b", "options"),
        Output("trial-b", "value"),
        Output("mtime-filter", "options"),
        Output("mtime-filter", "value"),
        Output("freq-min", "min"),
        Output("freq-min", "max"),
        Output("freq-max", "min"),
        Output("freq-max", "max"),
        Input("dataset-id", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        State("trial-a", "value"),
        State("trial-b", "value"),
        State("mtime-filter", "value"),
    )
    def sync_dataset_specific_controls(
        dataset_id: str,
        json_filename: str,
        csv_filename: str,
        current_trial_a: int | None,
        current_trial_b: int | None,
        current_mtime_filter: str | None,
    ):
        bundle = current_bundle(dataset_id, json_filename, csv_filename)
        trial_options = [{"label": str(trial_id), "value": trial_id} for trial_id in bundle.trial_ids]
        trial_a_value = current_trial_a if current_trial_a in bundle.trial_ids else bundle.trial_ids[0]
        default_trial_b = bundle.trial_ids[1] if len(bundle.trial_ids) > 1 else bundle.trial_ids[0]
        trial_b_value = current_trial_b if current_trial_b in bundle.trial_ids else default_trial_b
        mtime_values = sorted(bundle.subjects["mtime"].dropna().unique())
        mtime_options = [{"label": "All", "value": "All"}] + [{"label": label, "value": label} for label in mtime_values]
        mtime_filter_value = current_mtime_filter if current_mtime_filter in {"All", *mtime_values} else "All"
        positive_frequencies = [frequency for frequency in bundle.frequencies if frequency > 0]
        min_frequency = min(positive_frequencies) if positive_frequencies else min(bundle.frequencies)
        max_frequency = max(positive_frequencies) if positive_frequencies else max(bundle.frequencies)
        return (
            [{"label": bundle.metric, "value": bundle.metric}],
            bundle.metric,
            trial_options,
            trial_a_value,
            trial_options,
            trial_b_value,
            mtime_options,
            mtime_filter_value,
            min_frequency,
            max_frequency,
            min_frequency,
            max_frequency,
        )

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
        Output("excluded-idx-checklist", "options"),
        Output("patterns-color-variable", "options"),
        Output("patterns-color-variable", "value"),
        Output("patterns-behavior-variables", "options"),
        Output("patterns-behavior-variables", "value"),
        Input("dataset-id", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
    )
    def sync_file_dependent_controls(dataset_id: str, json_filename: str, csv_filename: str):
        bundle = current_bundle(dataset_id, json_filename, csv_filename)
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
        patterns_color_options = [
            {"label": "Group", "value": "group_label"},
            {"label": "Timepoint", "value": "timepoint_label"},
            {"label": "IDX", "value": "idx"},
        ] + numeric_options
        default_behavior = numeric_columns[:3]
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
            _build_exclusion_options(bundle),
            patterns_color_options,
            "group_label",
            numeric_options,
            default_behavior,
        )

    @app.callback(
        Output("excluded-idx-checklist", "value"),
        Input("dataset-id", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        Input("outlier-reset-exclusions", "n_clicks"),
        Input("outlier-top3-button", "n_clicks"),
        State("excluded-idx-checklist", "value"),
        State("excluded-idx-checklist", "options"),
        State("outlier-top3-store", "data"),
    )
    def update_excluded_idx_selection(
        dataset_id: str,
        json_filename: str,
        csv_filename: str,
        reset_clicks: int,
        top3_clicks: int,
        current_value: list[str] | None,
        options: list[dict[str, Any]] | None,
        top3_candidates: list[str] | None,
    ) -> list[str]:
        _ = current_bundle(dataset_id, json_filename, csv_filename)
        valid_values = {str(option["value"]) for option in (options or [])}
        current = [str(value) for value in (current_value or []) if str(value) in valid_values]
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if triggered in {"dataset-id", "json-file", "csv-file"}:
            return current
        if triggered == "outlier-reset-exclusions":
            return []
        if triggered == "outlier-top3-button":
            return sorted({*current, *(str(value) for value in (top3_candidates or []) if str(value) in valid_values)})
        return current

    @app.callback(
        Output("excluded-idx-store", "data"),
        Input("excluded-idx-checklist", "value"),
    )
    def sync_excluded_idx_store(selected_values: list[str] | None) -> list[str]:
        return sorted({str(value) for value in (selected_values or [])})

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
        Input("dataset-id", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        Input("longitudinal-enabled", "value"),
    )
    def sync_longitudinal_value_options(dataset_id: str, json_filename: str, csv_filename: str, enabled_flags: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
        if "enabled" not in (enabled_flags or []):
            return [], [], None, None
        bundle = current_bundle(dataset_id, json_filename, csv_filename)
        values = [value for value in bundle.subjects["mtime"].dropna().unique().tolist()]
        values = sorted(values, key=lambda item: str(item))
        options = [{"label": str(value), "value": value} for value in values]
        first = values[0] if values else None
        second = values[1] if len(values) > 1 else first
        return options, options, first, second

    @app.callback(
        Output("selected-pair", "data"),
        Input("heatmap", "clickData"),
        Input("dataset-id", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        State("selected-pair", "data"),
        prevent_initial_call=True,
    )
    def select_pair(click_data: dict[str, Any] | None, dataset_id: str, json_filename: str, csv_filename: str, current_pair: str) -> str:
        bundle = current_bundle(dataset_id, json_filename, csv_filename)
        default_pair = f"{bundle.channels[0]}|{bundle.channels[1]}"
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if triggered in {"dataset-id", "json-file", "csv-file"}:
            return default_pair
        if not click_data or not click_data.get("points"):
            return current_pair if current_pair else default_pair
        point = click_data["points"][0]
        roi_from = point.get("y")
        roi_to = point.get("x")
        if roi_from == roi_to or roi_from not in bundle.channels or roi_to not in bundle.channels:
            return current_pair if current_pair else default_pair
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
        Output("heatmap-longitudinal-summary", "children"),
        Output("heatmap-longitudinal-fit-chart", "figure"),
        Output("heatmap-longitudinal-residual-chart", "figure"),
        Output("heatmap-longitudinal-robustness-summary", "children"),
        Output("heatmap-longitudinal-trajectory-summary", "children"),
        Output("heatmap-longitudinal-trajectory-chart", "figure"),
        Output("heatmap-reliable-change-summary", "children"),
        Output("heatmap-reliable-change-chart", "figure"),
        Input("dataset-id", "value"),
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
        Input("heatmap-longitudinal-model", "value"),
        Input("heatmap-longitudinal-random-slope", "value"),
        Input("heatmap-longitudinal-show-trajectories", "value"),
        Input("heatmap-longitudinal-show-reliable-change", "value"),
        Input("excluded-idx-store", "data"),
        Input("selected-pair", "data"),
    )
    def update_dashboard(
        dataset_id: str,
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
        heatmap_longitudinal_model: str | None,
        heatmap_longitudinal_random_slope: list[str] | None,
        heatmap_longitudinal_show_trajectories: list[str] | None,
        heatmap_longitudinal_show_reliable_change: list[str] | None,
        excluded_idx: list[str] | None,
        selected_pair: str,
    ) -> tuple[Any, Any, go.Figure, go.Figure, Any, go.Figure, Any, Any, go.Figure, go.Figure, Any, go.Figure, go.Figure, Any, Any, go.Figure, Any, go.Figure]:
        bundle, _, analysis_result, effective_analysis_mode = resolve_analysis_context(
            dataset_id=dataset_id,
            json_filename=json_filename,
            csv_filename=csv_filename,
            group_a=group_a,
            group_b=group_b,
            longitudinal_enabled_flags=longitudinal_enabled_flags,
            longitudinal_pairing_flags=longitudinal_pairing_flags,
            longitudinal_value_a=longitudinal_value_a,
            longitudinal_value_b=longitudinal_value_b,
            trial_a=trial_a,
            trial_b=trial_b,
            band_preset=band_preset,
            freq_min=freq_min,
            freq_max=freq_max,
            mtime_filter=mtime_filter,
            correction_method=correction_method,
            excluded_idx=excluded_idx,
        )
        significance_threshold = float(significance_threshold) if significance_threshold is not None else 0.05
        significance_threshold = min(max(significance_threshold, 0.0), 1.0)
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
        longitudinal_detail_result = run_longitudinal_detail_analysis(
            bundle,
            analysis_result,
            selected_pair,
            config=LongitudinalDetailConfig(
                model_family=heatmap_longitudinal_model or "mixed_effects",
                random_slope_time="enabled" in (heatmap_longitudinal_random_slope or []),
                trajectory_ci=0.95,
                reliable_change_method="jt_baseline_sd",
                baseline_value=longitudinal_value_a,
                followup_value=longitudinal_value_b,
                reliability_coefficient=0.8,
            ),
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
                    f"{'paired repeated-subject analysis only' if 'paired_only' in (longitudinal_pairing_flags or []) else 'all available subjects at each selected timepoint'}"
                    if 'enabled' in (longitudinal_enabled_flags or [])
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
        longitudinal_summary, longitudinal_fit, longitudinal_residual, longitudinal_robustness, longitudinal_trajectory_summary, longitudinal_trajectory_chart, reliable_summary, reliable_chart = _build_heatmap_longitudinal_view(
            longitudinal_detail_result=longitudinal_detail_result,
            show_trajectories="enabled" in (heatmap_longitudinal_show_trajectories or []),
            show_reliable_change="enabled" in (heatmap_longitudinal_show_reliable_change or []),
        )
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
            longitudinal_summary,
            longitudinal_fit,
            longitudinal_residual,
            longitudinal_robustness,
            longitudinal_trajectory_summary,
            longitudinal_trajectory_chart,
            reliable_summary,
            reliable_chart,
        )

    @app.callback(
        Output("network-overview", "children"),
        Output("network-primary-summary", "children"),
        Output("network-primary-chart", "figure"),
        Output("network-secondary-summary", "children"),
        Output("network-secondary-chart", "figure"),
        Input("dataset-id", "value"),
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
        Input("network-mode", "value"),
        Input("network-metric-name", "value"),
        Input("network-weight-mode", "value"),
        Input("network-threshold-mode", "value"),
        Input("network-threshold-value", "value"),
        Input("network-nbs-primary-threshold", "value"),
        Input("network-nbs-permutations", "value"),
        Input("excluded-idx-store", "data"),
    )
    def update_network_dashboard(
        dataset_id: str,
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
        network_mode: str,
        network_metric_name: str,
        network_weight_mode: str,
        network_threshold_mode: str,
        network_threshold_value: float | None,
        network_nbs_primary_threshold: float | None,
        network_nbs_permutations: int | None,
        excluded_idx: list[str] | None,
    ) -> tuple[Any, Any, go.Figure, Any, go.Figure]:
        bundle, _, analysis_result, _ = resolve_analysis_context(
            dataset_id=dataset_id,
            json_filename=json_filename,
            csv_filename=csv_filename,
            group_a=group_a,
            group_b=group_b,
            longitudinal_enabled_flags=longitudinal_enabled_flags,
            longitudinal_pairing_flags=longitudinal_pairing_flags,
            longitudinal_value_a=longitudinal_value_a,
            longitudinal_value_b=longitudinal_value_b,
            trial_a=trial_a,
            trial_b=trial_b,
            band_preset=band_preset,
            freq_min=freq_min,
            freq_max=freq_max,
            mtime_filter=mtime_filter,
            correction_method=correction_method,
            excluded_idx=excluded_idx,
        )
        network_result = run_network_analysis(
            bundle,
            analysis_result,
            NetworkAnalysisConfig(
                mode=network_mode or "summary",
                metric_name=network_metric_name or "roi_mean_connectivity",
                weight_mode=network_weight_mode or "positive",
                threshold_mode=network_threshold_mode or "none",
                threshold_value=float(network_threshold_value) if network_threshold_value is not None else 0.0,
                network_labeling_mode="anatomical",
                nbs_primary_threshold=float(network_nbs_primary_threshold) if network_nbs_primary_threshold is not None else 2.0,
                nbs_permutations=int(network_nbs_permutations) if network_nbs_permutations is not None else 200,
            ),
        )
        return _build_network_view(
            network_result=network_result,
            network_mode=network_mode or "summary",
            metric_name=network_metric_name or "roi_mean_connectivity",
            weight_mode=network_weight_mode or "positive",
            threshold_mode=network_threshold_mode or "none",
            threshold_value=float(network_threshold_value) if network_threshold_value is not None else 0.0,
        )

    @app.callback(
        Output("patterns-overview", "children"),
        Output("patterns-primary-summary", "children"),
        Output("patterns-primary-chart", "figure"),
        Output("patterns-secondary-summary", "children"),
        Output("patterns-secondary-chart", "figure"),
        Input("dataset-id", "value"),
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
        Input("patterns-mode", "value"),
        Input("patterns-embedding-method", "value"),
        Input("patterns-embedding-dim", "value"),
        Input("patterns-color-variable", "value"),
        Input("patterns-cluster-method", "value"),
        Input("patterns-n-clusters", "value"),
        Input("patterns-feature-level", "value"),
        Input("patterns-brain-behavior-method", "value"),
        Input("patterns-behavior-variables", "value"),
        Input("patterns-n-components", "value"),
        Input("patterns-standardize", "value"),
        Input("excluded-idx-store", "data"),
    )
    def update_patterns_dashboard(
        dataset_id: str,
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
        patterns_mode: str,
        patterns_embedding_method: str,
        patterns_embedding_dim: int | None,
        patterns_color_variable: str | None,
        patterns_cluster_method: str,
        patterns_n_clusters: int | None,
        patterns_feature_level: str,
        patterns_brain_behavior_method: str,
        patterns_behavior_variables: list[str] | None,
        patterns_n_components: int | None,
        patterns_standardize: str,
        excluded_idx: list[str] | None,
    ) -> tuple[Any, Any, go.Figure, Any, go.Figure]:
        bundle, _, analysis_result, _ = resolve_analysis_context(
            dataset_id=dataset_id,
            json_filename=json_filename,
            csv_filename=csv_filename,
            group_a=group_a,
            group_b=group_b,
            longitudinal_enabled_flags=longitudinal_enabled_flags,
            longitudinal_pairing_flags=longitudinal_pairing_flags,
            longitudinal_value_a=longitudinal_value_a,
            longitudinal_value_b=longitudinal_value_b,
            trial_a=trial_a,
            trial_b=trial_b,
            band_preset=band_preset,
            freq_min=freq_min,
            freq_max=freq_max,
            mtime_filter=mtime_filter,
            correction_method=correction_method,
            excluded_idx=excluded_idx,
        )
        patterns_result = run_patterns_analysis(
            bundle,
            analysis_result,
            PatternsAnalysisConfig(
                mode=patterns_mode or "embedding",
                standardize=patterns_standardize or "zscore",
                embedding_method=patterns_embedding_method or "pca",
                embedding_dim=int(patterns_embedding_dim) if patterns_embedding_dim is not None else 2,
                color_variable=patterns_color_variable or "group_label",
                cluster_method=patterns_cluster_method or "kmeans",
                n_clusters=int(patterns_n_clusters) if patterns_n_clusters is not None else 3,
                feature_pattern_level=patterns_feature_level or "edges",
                cca_pls_method=patterns_brain_behavior_method or "pls",
                behavior_variables=tuple(patterns_behavior_variables or ()),
                n_components=int(patterns_n_components) if patterns_n_components is not None else 2,
            ),
        )
        return _build_patterns_view(
            patterns_result=patterns_result,
            patterns_mode=patterns_mode or "embedding",
            color_variable=patterns_color_variable or "group_label",
        )

    @app.callback(
        Output("heatmap-status", "children"),
        Input("heatmap", "loading_state"),
        Input("heatmap-correlation-chart", "loading_state"),
        Input("heatmap-regression-fit-chart", "loading_state"),
        Input("heatmap-longitudinal-fit-chart", "loading_state"),
        Input("heatmap-longitudinal-trajectory-chart", "loading_state"),
        Input("heatmap-reliable-change-chart", "loading_state"),
    )
    def update_heatmap_status(*loading_states: dict[str, Any] | None) -> str:
        return _loading_status_text("Heatmap", loading_states)

    @app.callback(
        Output("network-status", "children"),
        Input("network-primary-chart", "loading_state"),
        Input("network-secondary-chart", "loading_state"),
        Input("network-overview", "loading_state"),
    )
    def update_network_status(*loading_states: dict[str, Any] | None) -> str:
        return _loading_status_text("Network", loading_states)

    @app.callback(
        Output("patterns-status", "children"),
        Input("patterns-primary-chart", "loading_state"),
        Input("patterns-secondary-chart", "loading_state"),
        Input("patterns-overview", "loading_state"),
    )
    def update_patterns_status(*loading_states: dict[str, Any] | None) -> str:
        return _loading_status_text("Patterns", loading_states)

    @app.callback(
        Output("outlier-status", "children"),
        Input("outlier-global-chart", "loading_state"),
        Input("outlier-pair-chart", "loading_state"),
        Input("outlier-regression-cook-chart", "loading_state"),
    )
    def update_outlier_status(*loading_states: dict[str, Any] | None) -> str:
        return _loading_status_text("Outlier", loading_states)

    @app.callback(
        Output("outlier-top3-store", "data"),
        Output("outlier-overview", "children"),
        Output("outlier-global-chart", "figure"),
        Output("outlier-pair-summary", "children"),
        Output("outlier-pair-chart", "figure"),
        Output("outlier-regression-summary", "children"),
        Output("outlier-regression-leverage-chart", "figure"),
        Output("outlier-regression-cook-chart", "figure"),
        Input("main-tabs", "value"),
        Input("dataset-id", "value"),
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
        Input("significance-threshold", "value"),
        Input("regression-covariates", "value"),
        Input("heatmap-regression-outcome-variable", "value"),
        Input("excluded-idx-store", "data"),
        Input("selected-pair", "data"),
        Input("outlier-mode", "value"),
    )
    def update_outlier_dashboard(
        active_tab: str,
        dataset_id: str,
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
        significance_threshold: float,
        regression_covariates: list[str] | None,
        heatmap_regression_outcome_variable: str | None,
        excluded_idx: list[str] | None,
        selected_pair: str,
        outlier_mode: str,
    ) -> tuple[list[str], Any, go.Figure, Any, go.Figure, Any, go.Figure, go.Figure]:
        if active_tab != "outlier":
            placeholder = html.P("Open the Outlier tab to compute leave-one-out and influence diagnostics.")
            empty = _empty_figure("Outlier analysis")
            return [], placeholder, empty, placeholder, empty, placeholder, empty, empty

        bundle = current_bundle(dataset_id, json_filename, csv_filename)
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
            excluded_idx=tuple(sorted({str(value) for value in (excluded_idx or [])})),
        )

        outlier_result = run_leave_one_out_analysis(
            bundle,
            config,
            selected_pair,
            significance_threshold=significance_threshold,
            outcome_variable=heatmap_regression_outcome_variable,
            regression_covariates=regression_covariates,
        )
        regression_influence = run_regression_influence_analysis(
            bundle,
            outlier_result["baseline_analysis"],
            selected_pair,
            outcome_variable=heatmap_regression_outcome_variable,
            regression_covariates=regression_covariates,
        )

        overview = _build_outlier_overview(
            outlier_result,
            selected_pair=selected_pair,
            excluded_idx=excluded_idx or [],
            outlier_mode=outlier_mode,
        )
        global_component, global_figure = _build_global_influence_view(outlier_result) if outlier_mode == "global" else (html.P("Switch the outlier view to 'Global influence' to inspect global subject impact."), _empty_figure("Global influence"))
        pair_component, pair_figure = _build_pair_influence_view(outlier_result) if outlier_mode == "pair" else (html.P("Switch the outlier view to 'Selected ROI pair' to inspect pair-wise influence."), _empty_figure("Selected ROI pair influence"))
        regression_component, leverage_figure, cook_figure = (
            _build_regression_influence_view(regression_influence, outlier_result["regression_records"])
            if outlier_mode == "regression"
            else (
                html.P("Switch the outlier view to 'Regression diagnostics' to inspect leverage, studentized residuals, and Cook's distance."),
                _empty_figure("Leverage vs studentized residuals"),
                _empty_figure("Cook's distance"),
            )
        )

        return (
            outlier_result["top3_idx"],
            html.Div([overview, global_component] if outlier_mode == "global" else [overview]),
            global_figure,
            pair_component,
            pair_figure,
            regression_component,
            leverage_figure,
            cook_figure,
        )

    return app


def _sidebar(
    bundle: DatasetBundle,
    dataset_ids: list[str],
    initial_dataset: str,
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
            dcc.Dropdown(
                id="dataset-id",
                options=[{"label": dataset_id, "value": dataset_id} for dataset_id in dataset_ids],
                value=initial_dataset,
                clearable=False,
            ),
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
            dcc.Dropdown(id="metric-name", options=[{"label": bundle.metric, "value": bundle.metric}], value=bundle.metric, disabled=True),
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
    point_labels = [label for label, keep in zip(point_labels, mask.tolist()) if keep] if point_labels else []
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


def _build_heatmap_longitudinal_view(
    *,
    longitudinal_detail_result: dict[str, Any],
    show_trajectories: bool,
    show_reliable_change: bool,
) -> tuple[Any, go.Figure, go.Figure, Any, Any, go.Figure, Any, go.Figure]:
    fit_figure = go.Figure()
    residual_figure = go.Figure()
    trajectory_figure = go.Figure()
    reliable_figure = go.Figure()

    if longitudinal_detail_result.get("message"):
        fit_figure.update_layout(title="Longitudinal observed vs fitted", height=360)
        residual_figure.update_layout(title="Longitudinal residuals vs fitted", height=360)
        trajectory_figure.update_layout(title="Selected-edge trajectories", height=420)
        reliable_figure.update_layout(title="Reliable change", height=360)
        message = html.P(longitudinal_detail_result["message"])
        return message, fit_figure, residual_figure, message, message, trajectory_figure, message, reliable_figure

    primary_model = longitudinal_detail_result.get("primary_model", {})
    summary, fit_figure, residual_figure = _build_longitudinal_model_view(primary_model)
    robustness = _build_longitudinal_robustness_summary(
        longitudinal_detail_result.get("change_score", {}),
        longitudinal_detail_result.get("ancova", {}),
    )

    if show_trajectories:
        trajectory_summary, trajectory_figure = _build_longitudinal_trajectory_view(longitudinal_detail_result.get("trajectory", {}))
    else:
        trajectory_summary = html.P("Trajectory display is disabled.")
        trajectory_figure.update_layout(title="Selected-edge trajectories", height=420)

    if show_reliable_change:
        reliable_summary, reliable_figure = _build_reliable_change_view(longitudinal_detail_result.get("reliable_change", {}))
    else:
        reliable_summary = html.P("Reliable change display is disabled.")
        reliable_figure.update_layout(title="Reliable change", height=360)

    return summary, fit_figure, residual_figure, robustness, trajectory_summary, trajectory_figure, reliable_summary, reliable_figure


def _build_longitudinal_model_view(model_result: dict[str, Any]) -> tuple[Any, go.Figure, go.Figure]:
    fit_figure = go.Figure()
    residual_figure = go.Figure()
    if model_result.get("message"):
        fit_figure.update_layout(title="Longitudinal observed vs fitted", height=360)
        residual_figure.update_layout(title="Longitudinal residuals vs fitted", height=360)
        return html.P(model_result["message"]), fit_figure, residual_figure

    summary_lines = [
        html.H4("Longitudinal model"),
        html.P(model_result["model_label"]),
        html.P(f"Formula: {model_result['formula']}"),
        html.P(f"Response: {model_result['response_definition']}"),
        html.P(f"Observation unit: {model_result['observation_unit']}"),
        html.P(model_result["methods_description"]),
        html.P(f"n = {model_result['n']}"),
    ]
    if "r_squared" in model_result:
        summary_lines.append(html.P(f"R² = {model_result['r_squared']:.6f}, adjusted R² = {model_result['adj_r_squared']:.6f}"))
    primary = model_result.get("primary_effect")
    if primary:
        summary_lines.append(
            html.P(
                f"Primary longitudinal effect: {primary['label']} | estimate = {primary['estimate']:.6f}, "
                f"t/z = {primary['t_value']:.6f}, p = {primary['p_value']:.6g}"
            )
        )
        if primary.get("interpretation"):
            summary_lines.append(html.P(primary["interpretation"]))
    variance_components = model_result.get("variance_components")
    if variance_components:
        rows = []
        for label, value in variance_components.items():
            if value == value:
                rows.append(html.Li(f"{label.replace('_', ' ')} = {value:.6f}"))
        if rows:
            summary_lines.append(html.Ul(rows))
    diagnostics = model_result.get("diagnostics", {})
    if diagnostics.get("message"):
        summary_lines.append(html.P(diagnostics["message"]))

    coefficient_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Term", style=_stats_cell_style(header=True)),
                        html.Th("Estimate", style=_stats_cell_style(header=True)),
                        html.Th("SE", style=_stats_cell_style(header=True)),
                        html.Th("t / z", style=_stats_cell_style(header=True)),
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
                    for row in model_result.get("coefficients", [])
                ]
            ),
        ],
        style=_stats_table_style(),
    )

    observed = diagnostics.get("observed", [])
    fitted = diagnostics.get("fitted", [])
    residuals = diagnostics.get("residuals", [])
    point_labels = diagnostics.get("point_labels")

    fit_figure.add_trace(
        go.Scatter(
            x=fitted,
            y=observed,
            mode="markers",
            text=point_labels,
            marker={"size": 8, "color": "#2a6f97", "opacity": 0.85},
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
    fit_figure.update_layout(title="Longitudinal observed vs fitted", xaxis_title="Fitted", yaxis_title="Observed", height=360)

    residual_figure.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            text=point_labels,
            marker={"size": 8, "color": "#ef8354", "opacity": 0.85},
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
    residual_figure.update_layout(title="Longitudinal residuals vs fitted", xaxis_title="Fitted", yaxis_title="Residual", height=360)
    return html.Div(summary_lines + [coefficient_table]), fit_figure, residual_figure


def _build_longitudinal_robustness_summary(change_score_result: dict[str, Any], ancova_result: dict[str, Any]) -> Any:
    def _card(title: str, result: dict[str, Any]) -> Any:
        if result.get("message"):
            return html.Div([html.Div(title, style=_stats_section_title_style()), html.P(result["message"])], className="panel")
        primary = result.get("primary_effect")
        rows = [
            html.Tr([html.Td("Model", style=_stats_cell_style(label=True)), html.Td(result["model_label"], style=_stats_cell_style())]),
            html.Tr([html.Td("n", style=_stats_cell_style(label=True)), html.Td(str(result["n"]), style=_stats_cell_style())]),
            html.Tr([html.Td("R²", style=_stats_cell_style(label=True)), html.Td(f"{result['r_squared']:.6f}", style=_stats_cell_style())]),
        ]
        if primary:
            rows.extend(
                [
                    html.Tr([html.Td("Primary effect", style=_stats_cell_style(label=True)), html.Td(primary["label"], style=_stats_cell_style())]),
                    html.Tr([html.Td("Estimate", style=_stats_cell_style(label=True)), html.Td(f"{primary['estimate']:.6f}", style=_stats_cell_style())]),
                    html.Tr([html.Td("p", style=_stats_cell_style(label=True)), html.Td(f"{primary['p_value']:.6g}", style=_stats_cell_style())]),
                ]
            )
        return html.Div(
            [
                html.Div(title, style=_stats_section_title_style()),
                html.Table([html.Tbody(rows)], style=_stats_table_style()),
            ],
            className="panel",
        )

    return html.Div(
        [
            html.H4("Longitudinal robustness"),
            html.P("Change-score and ANCOVA are shown in parallel so the selected-edge longitudinal result can be checked for robustness."),
            html.Div(
                [
                    _card("Change score", change_score_result),
                    _card("ANCOVA", ancova_result),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))", "gap": "14px"},
            ),
        ]
    )


def _build_longitudinal_trajectory_view(trajectory_result: dict[str, Any]) -> tuple[Any, go.Figure]:
    figure = go.Figure()
    if trajectory_result.get("message"):
        figure.update_layout(title="Selected-edge trajectories", height=420)
        return html.P(trajectory_result["message"]), figure

    ordered_labels = trajectory_result.get("ordered_timepoints", [])
    palette = ["rgba(31,119,180,0.18)", "rgba(255,127,14,0.18)", "rgba(44,160,44,0.18)", "rgba(148,103,189,0.18)"]
    trend_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for trace in trajectory_result.get("subject_traces", []):
        figure.add_trace(
            go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode="lines+markers",
                line={"color": "#c7cbd1", "width": 1},
                marker={"size": 5, "color": "#c7cbd1"},
                name=f"IDX {trace['idx']}",
                legendgroup=f"subject-{trace['idx']}",
                showlegend=False,
                hovertemplate=f"IDX={trace['idx']}<br>Group={trace['group_label']}<br>Time=%{{x}}<br>Delta=%{{y:.4f}}<extra></extra>",
            )
        )

    for index, group in enumerate(trajectory_result.get("group_trends", [])):
        rows = group.get("rows", [])
        x = [row["timepoint_label"] for row in rows]
        mean = [row["mean"] for row in rows]
        lower = [row["lower"] for row in rows]
        upper = [row["upper"] for row in rows]
        line_color = trend_palette[index % len(trend_palette)]
        fill_color = palette[index % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=fill_color,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name=f"{group['group_label']} 95% CI",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines+markers",
                line={"color": line_color, "width": 3},
                marker={"size": 8, "color": line_color},
                name=f"{group['group_label']} mean",
                hovertemplate=f"Group={group['group_label']}<br>Time=%{{x}}<br>Mean delta=%{{y:.4f}}<extra></extra>",
            )
        )

    figure.update_layout(
        title="Selected-edge trajectories",
        xaxis={"title": "Timepoint", "categoryorder": "array", "categoryarray": ordered_labels},
        yaxis_title="Trial delta",
        height=420,
        legend={"orientation": "h", "x": 0, "y": 1.12},
    )

    summary = html.Div(
        [
            html.H4("Longitudinal trajectories"),
            html.P("Individual subject lines are shown in light gray; colored lines show group means with confidence bands."),
            html.P(
                "The plot uses all available timepoints for the selected edge as context, even if the main inferential contrast currently targets a selected timepoint pair."
                if trajectory_result.get("uses_all_available_timepoints")
                else "The plot uses the selected timepoints only."
            ),
        ]
    )
    return summary, figure


def _build_reliable_change_view(reliable_change_result: dict[str, Any]) -> tuple[Any, go.Figure]:
    figure = go.Figure()
    if reliable_change_result.get("message"):
        figure.update_layout(title="Reliable change", height=360)
        return html.P(reliable_change_result["message"]), figure

    records = reliable_change_result.get("records", [])
    categories = ["reliable increase", "no reliable change", "reliable decrease"]
    counts = [reliable_change_result["category_counts"].get(category, 0) for category in categories]
    colors = ["#2ca02c", "#9aa0a6", "#d62728"]
    figure.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            marker={"color": colors},
            hovertemplate="Category=%{x}<br>n=%{y}<extra></extra>",
        )
    )
    figure.update_layout(title="Reliable change categories", xaxis_title="Category", yaxis_title="Count", height=360)

    subject_rows = [
        html.Tr(
            [
                html.Td(record["idx"], style=_stats_cell_style(label=True)),
                html.Td(record["group_label"], style=_stats_cell_style()),
                html.Td(f"{record['change_score']:.6f}", style=_stats_cell_style()),
                html.Td(f"{record['rci']:.6f}", style=_stats_cell_style()),
                html.Td(record["category"], style=_stats_cell_style()),
            ]
        )
        for record in records
    ]
    subject_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("IDX", style=_stats_cell_style(header=True)),
                        html.Th("Group", style=_stats_cell_style(header=True)),
                        html.Th("Change", style=_stats_cell_style(header=True)),
                        html.Th("RCI", style=_stats_cell_style(header=True)),
                        html.Th("Category", style=_stats_cell_style(header=True)),
                    ]
                )
            ),
            html.Tbody(subject_rows),
        ],
        style=_stats_table_style(),
    )
    summary = html.Div(
        [
            html.H4("Reliable change"),
            html.P(
                f"{reliable_change_result['method_label']} (assumed reliability = {reliable_change_result['reliability_coefficient']:.2f}, "
                f"baseline SD = {reliable_change_result['baseline_sd']:.6f})."
            ),
            subject_table,
        ]
    )
    return summary, figure


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


def _loading_status_text(section_name: str, loading_states: tuple[dict[str, Any] | None, ...]) -> str:
    if any(state and state.get("is_loading") for state in loading_states):
        return f"{section_name}: estimating ... please wait."
    return f"{section_name}: ready."


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title, height=420)
    return figure


def _build_exclusion_options(bundle: DatasetBundle) -> list[dict[str, str]]:
    options = []
    grouped = bundle.subjects.groupby("idx", sort=True)
    for idx, frame in grouped:
        idx = str(idx)
        groups = ", ".join(sorted(frame["group_label"].dropna().astype(str).unique().tolist()))
        mtimes = ", ".join(sorted(frame["mtime"].dropna().astype(str).unique().tolist()))
        subject_count = frame["subject_id"].nunique() if "subject_id" in frame.columns else len(frame)
        label = f"IDX {idx} | groups: {groups or 'n/a'} | mtime: {mtimes or 'n/a'} | measurements: {subject_count}"
        options.append({"label": label, "value": idx})
    return options


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_simple_table(rows: list[tuple[str, str]]) -> html.Table:
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


def _build_outlier_overview(
    outlier_result: dict[str, Any],
    *,
    selected_pair: str,
    excluded_idx: list[str],
    outlier_mode: str,
) -> html.Div:
    baseline_analysis = outlier_result["baseline_analysis"]
    top3 = outlier_result.get("top3_idx", [])
    return html.Div(
        [
            html.H4("Outlier analysis"),
            html.P(f"Selected pair: {selected_pair}"),
            html.P(f"Current mode: {outlier_mode}"),
            html.P(f"Analysis: {baseline_analysis['analysis_label']}"),
            html.P(f"Global exclusions ({len(excluded_idx)}): {', '.join(sorted(excluded_idx)) if excluded_idx else 'none'}"),
            html.P(f"Top 3 influential IDX: {', '.join(top3) if top3 else 'not available'}"),
        ]
    )


def _build_global_influence_view(outlier_result: dict[str, Any]) -> tuple[Any, go.Figure]:
    records = outlier_result.get("global_records", [])
    if not records:
        return html.P("No global influence records are available."), _empty_figure("Global influence")

    frame = pd.DataFrame(records)
    summary = html.Div(
        [
            html.H4("Global influence ranking"),
            html.P("Scores summarize how strongly the full heatmap changes when one IDX is removed."),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("IDX", style=_stats_cell_style(header=True)),
                                html.Th("Sig. loss", style=_stats_cell_style(header=True)),
                                html.Th("Sig. gain", style=_stats_cell_style(header=True)),
                                html.Th("Sig. switches", style=_stats_cell_style(header=True)),
                                html.Th("Mean |effect size change|", style=_stats_cell_style(header=True)),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(str(row["excluded_idx"]), style=_stats_cell_style(label=True)),
                                    html.Td(str(int(row["significance_loss_count"])), style=_stats_cell_style()),
                                    html.Td(str(int(row["significance_gain_count"])), style=_stats_cell_style()),
                                    html.Td(str(int(row["significance_switch_count"])), style=_stats_cell_style()),
                                    html.Td(_format_float(row["mean_abs_effect_size_change"]), style=_stats_cell_style()),
                                ]
                            )
                            for _, row in frame.iterrows()
                        ]
                    ),
                ],
                style=_stats_table_style(),
            ),
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["excluded_idx"].tolist(),
            y=frame["significance_switch_count"].astype(int).tolist(),
            name="Significance switches",
            marker={"color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["excluded_idx"].tolist(),
            y=frame["mean_abs_effect_size_change"].astype(float).tolist(),
            name="Mean |effect size change|",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#d62728", "width": 2},
        )
    )
    figure.update_layout(
        title="Global subject influence",
        xaxis_title="Excluded IDX",
        yaxis_title="Significance switches",
        yaxis2={"title": "Mean |effect size change|", "overlaying": "y", "side": "right"},
        height=460,
        legend={"orientation": "h", "x": 0, "y": 1.12},
    )
    return summary, figure


def _build_pair_influence_view(outlier_result: dict[str, Any]) -> tuple[Any, go.Figure]:
    pair_records = outlier_result.get("pair_records", [])
    regression_records = {row["excluded_idx"]: row for row in outlier_result.get("regression_records", [])}
    baseline_pair = outlier_result.get("baseline_pair")
    if not baseline_pair or not pair_records:
        return html.P("No leave-one-out records are available for the selected ROI pair."), _empty_figure("Selected ROI pair influence")

    table_rows = []
    for record in pair_records:
        regression_row = regression_records.get(record["excluded_idx"], {})
        coefficient_changes = regression_row.get("coefficient_changes", [])
        first_terms = ", ".join(
            f"{item['label']}: {_format_float(item['change'])}"
            for item in coefficient_changes[:3]
        ) or "n/a"
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(record["excluded_idx"]), style=_stats_cell_style(label=True)),
                    html.Td(_format_float(record["mean_delta_change"]), style=_stats_cell_style()),
                    html.Td(_format_float(record["effect_size_change"]), style=_stats_cell_style()),
                    html.Td(_format_float(record["p_value_change"]), style=_stats_cell_style()),
                    html.Td(_format_float(record["q_value_change"]), style=_stats_cell_style()),
                    html.Td("yes" if record["significance_switched"] else "no", style=_stats_cell_style()),
                    html.Td(_format_float(regression_row.get("primary_effect_change")), style=_stats_cell_style()),
                    html.Td(first_terms, style=_stats_cell_style()),
                ]
            )
        )

    summary = html.Div(
        [
            html.H4(f"Leave-one-out for {baseline_pair['roi_from']} -> {baseline_pair['roi_to']}"),
            html.P(
                f"Baseline: mean delta = {_format_float(baseline_pair['mean_delta'])}, "
                f"effect size = {_format_float(baseline_pair['effect_size'])}, "
                f"p = {_format_float(baseline_pair['p_value'])}, q = {_format_float(baseline_pair['q_value'])}"
            ),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Excluded IDX", style=_stats_cell_style(header=True)),
                                html.Th("Delta mean", style=_stats_cell_style(header=True)),
                                html.Th("Delta effect", style=_stats_cell_style(header=True)),
                                html.Th("Delta p", style=_stats_cell_style(header=True)),
                                html.Th("Delta q", style=_stats_cell_style(header=True)),
                                html.Th("Sig. switch", style=_stats_cell_style(header=True)),
                                html.Th("Primary coef.", style=_stats_cell_style(header=True)),
                                html.Th("Coefficient changes", style=_stats_cell_style(header=True)),
                            ]
                        )
                    ),
                    html.Tbody(table_rows),
                ],
                style=_stats_table_style(),
            ),
        ]
    )

    top_records = pair_records[:12]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[row["excluded_idx"] for row in top_records],
            y=[row["abs_mean_delta_change"] for row in top_records],
            name="|Mean delta change|",
            marker={"color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Bar(
            x=[row["excluded_idx"] for row in top_records],
            y=[row["abs_effect_size_change"] for row in top_records],
            name="|Effect size change|",
            marker={"color": "#ff7f0e"},
        )
    )
    figure.add_trace(
        go.Bar(
            x=[row["excluded_idx"] for row in top_records],
            y=[row["abs_q_value_change"] for row in top_records],
            name="|Q-value change|",
            marker={"color": "#2ca02c"},
        )
    )
    figure.update_layout(
        title="Influence ranking for the selected ROI pair",
        xaxis_title="Excluded IDX",
        yaxis_title="Absolute change",
        barmode="group",
        height=460,
    )
    return summary, figure


def _build_regression_influence_view(
    regression_influence: dict[str, Any],
    regression_records: list[dict[str, Any]],
) -> tuple[Any, go.Figure, go.Figure]:
    leverage_figure = _empty_figure("Leverage vs studentized residuals")
    cook_figure = _empty_figure("Cook's distance")
    if regression_influence.get("message"):
        return html.P(regression_influence["message"]), leverage_figure, cook_figure

    diagnostics = regression_influence.get("diagnostics", {})
    rows = diagnostics.get("observation_rows", [])
    if not rows:
        return html.P("No regression diagnostics are available."), leverage_figure, cook_figure

    frame = pd.DataFrame(rows)
    summary = html.Div(
        [
            html.H4("Regression influence diagnostics"),
            html.P(regression_influence["model_label"]),
            html.P(f"Formula: {regression_influence['formula']}"),
            html.P(f"Response: {regression_influence['response_definition']}"),
            html.P(f"n = {regression_influence['n']}"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Point", style=_stats_cell_style(header=True)),
                                html.Th("IDX", style=_stats_cell_style(header=True)),
                                html.Th("Residual", style=_stats_cell_style(header=True)),
                                html.Th("Studentized residual", style=_stats_cell_style(header=True)),
                                html.Th("Leverage", style=_stats_cell_style(header=True)),
                                html.Th("Cook's D", style=_stats_cell_style(header=True)),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(str(row.get("label", "n/a")), style=_stats_cell_style(label=True)),
                                    html.Td(str(row.get("idx", row.get("base_subject_id", "n/a"))), style=_stats_cell_style()),
                                    html.Td(_format_float(row.get("residual")), style=_stats_cell_style()),
                                    html.Td(_format_float(row.get("studentized_residual")), style=_stats_cell_style()),
                                    html.Td(_format_float(row.get("leverage")), style=_stats_cell_style()),
                                    html.Td(_format_float(row.get("cooks_distance")), style=_stats_cell_style()),
                                ]
                            )
                            for row in rows
                        ]
                    ),
                ],
                style=_stats_table_style(),
            ),
            html.Div(
                [
                    html.Div("Leave-one-out coefficient drift", style=_stats_section_title_style()),
                    _build_simple_table(
                        [
                            (
                                record["excluded_idx"],
                                f"Primary effect change = {_format_float(record.get('primary_effect_change'))}; "
                                f"R² change = {_format_float(record.get('r_squared_change'))}",
                            )
                            for record in regression_records[:8]
                        ]
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ]
    )

    leverage_figure = go.Figure()
    leverage_figure.add_trace(
        go.Scatter(
            x=frame["leverage"].astype(float).tolist(),
            y=frame["studentized_residual"].astype(float).tolist(),
            text=frame["label"].astype(str).tolist(),
            mode="markers+text",
            textposition="top center",
            marker={"size": 9, "color": "#1f77b4", "opacity": 0.8},
            hovertemplate="Point=%{text}<br>Leverage=%{x:.4f}<br>Studentized residual=%{y:.4f}<extra></extra>",
        )
    )
    leverage_figure.update_layout(
        title="Leverage vs studentized residuals",
        xaxis_title="Leverage",
        yaxis_title="Studentized residual",
        height=440,
    )

    order = frame.sort_values("cooks_distance", ascending=False)
    cook_figure = go.Figure()
    cook_figure.add_trace(
        go.Bar(
            x=order["label"].astype(str).tolist(),
            y=order["cooks_distance"].astype(float).tolist(),
            marker={"color": "#d62728"},
            hovertemplate="Point=%{x}<br>Cook's D=%{y:.4f}<extra></extra>",
        )
    )
    cook_figure.update_layout(
        title="Cook's distance",
        xaxis_title="Observation",
        yaxis_title="Cook's D",
        height=440,
    )
    return summary, leverage_figure, cook_figure


def _build_network_view(
    *,
    network_result: dict[str, Any],
    network_mode: str,
    metric_name: str,
    weight_mode: str,
    threshold_mode: str,
    threshold_value: float,
) -> tuple[Any, Any, go.Figure, Any, go.Figure]:
    overview = html.Div(
        [
            html.H4("Network analysis"),
            html.P(f"Mode: {network_mode}"),
            html.P(f"Metric family: {metric_name}"),
            html.P(f"Weight mode: {weight_mode}"),
            html.P(f"Threshold: {threshold_mode} ({threshold_value:.3f})"),
            html.P(f"Observations: {network_result.get('observation_count', 0)}"),
        ]
    )
    if network_mode == "graph":
        return (overview, *_build_graph_network_view(network_result, metric_name))
    if network_mode == "modules":
        return (overview, *_build_modules_network_view(network_result))
    if network_mode == "nbs":
        return (overview, *_build_nbs_network_view(network_result))
    return (overview, *_build_summary_network_view(network_result, metric_name))


def _build_patterns_view(
    *,
    patterns_result: dict[str, Any],
    patterns_mode: str,
    color_variable: str,
) -> tuple[Any, Any, go.Figure, Any, go.Figure]:
    feature_data = patterns_result.get("feature_matrix", {})
    metadata = pd.DataFrame(feature_data.get("metadata", []))
    overview = html.Div(
        [
            html.H4("Patterns analysis"),
            html.P(f"Mode: {patterns_mode}"),
            html.P(f"Observations: {len(metadata)}"),
            html.P(f"Features: {len(feature_data.get('feature_labels', []))}"),
            html.P(f"Color variable: {color_variable}"),
        ]
    )
    if patterns_result.get("message"):
        placeholder = html.P(patterns_result["message"])
        return overview, placeholder, _empty_figure("Patterns"), placeholder, _empty_figure("Patterns detail")
    if patterns_mode == "subject_clusters":
        return overview, *_build_patterns_clusters_view(patterns_result, color_variable)
    if patterns_mode == "feature_patterns":
        return overview, *_build_patterns_feature_view(patterns_result)
    if patterns_mode == "brain_behavior":
        return overview, *_build_patterns_brain_behavior_view(patterns_result)
    return overview, *_build_patterns_embedding_view(patterns_result, color_variable)


def _build_patterns_embedding_view(patterns_result: dict[str, Any], color_variable: str) -> tuple[Any, go.Figure, Any, go.Figure]:
    embedding = patterns_result.get("embedding_results", {})
    if embedding.get("message"):
        placeholder = html.P(embedding["message"])
        return placeholder, _empty_figure("Embedding"), placeholder, _empty_figure("PCA loadings")
    metadata = pd.DataFrame(embedding.get("metadata", []))
    dim = int(embedding.get("dim", 2))
    color_values = metadata[color_variable] if color_variable in metadata.columns else metadata.get("group_label", pd.Series(["n/a"] * len(metadata)))
    figure = go.Figure()
    if dim >= 3:
        figure.add_trace(
            go.Scatter3d(
                x=metadata["component_1"],
                y=metadata["component_2"],
                z=metadata["component_3"],
                mode="markers",
                marker={"size": 7, "color": _coerce_color_values(color_values)},
                text=_build_patterns_hover_labels(metadata),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=metadata["component_1"],
                y=metadata["component_2"],
                mode="markers",
                marker={"size": 9, "color": _coerce_color_values(color_values)},
                text=_build_patterns_hover_labels(metadata),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"{embedding['method'].upper()} subject embedding",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=500,
    )
    explained = embedding.get("explained_variance_ratio", [])
    summary = html.Div(
        [
            html.P(f"Method: {embedding['method'].upper()}"),
            html.P("Explained variance ratio: " + ", ".join(f"PC{i + 1}={value:.3f}" for i, value in enumerate(explained)) if explained else "Explained variance ratio: n/a"),
            _build_patterns_loading_table(pd.DataFrame(embedding.get("top_feature_loadings", []))),
        ]
    )
    loading_rows = embedding.get("top_feature_loadings", [])
    loading_figure = go.Figure()
    if loading_rows:
        first = loading_rows[0]["features"]
        loading_figure.add_trace(go.Bar(x=[row["feature"] for row in first], y=[row["loading"] for row in first]))
    loading_figure.update_layout(title="Top loadings for component 1", xaxis_title="Feature", yaxis_title="Loading", height=420)
    return summary, figure, html.P("Strongest contributing edges to the first PCA component"), loading_figure


def _build_patterns_clusters_view(patterns_result: dict[str, Any], color_variable: str) -> tuple[Any, go.Figure, Any, go.Figure]:
    clustering = patterns_result.get("subject_cluster_results", {})
    if clustering.get("message"):
        placeholder = html.P(clustering["message"])
        return placeholder, _empty_figure("Subject clusters"), placeholder, _empty_figure("Cluster summary")
    metadata = pd.DataFrame(clustering.get("metadata", []))
    overlay = pd.DataFrame(clustering.get("embedding_overlay", []))
    if not overlay.empty and "observation_id" in metadata.columns and "observation_id" in overlay.columns:
        overlay_columns = [column for column in overlay.columns if column.startswith("component_")] + ["observation_id"]
        metadata = metadata.merge(overlay[overlay_columns], on="observation_id", how="left", validate="one_to_one")
    summary = html.Div(
        [
            html.P(f"Method: {clustering['method']}"),
            html.P(f"Clusters: {clustering['n_clusters']}"),
            _build_patterns_cluster_summary_table(pd.DataFrame(clustering.get("cluster_summary", []))),
        ]
    )
    figure = go.Figure()
    if {"component_1", "component_2"}.issubset(metadata.columns):
        figure.add_trace(
            go.Scatter(
                x=metadata["component_1"],
                y=metadata["component_2"],
                mode="markers+text",
                marker={"size": 9, "color": metadata["cluster_label"]},
                text=metadata["cluster_label"].astype(str),
                textposition="top center",
                hovertext=_build_patterns_hover_labels(metadata),
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    figure.update_layout(title="Subject clusters in embedding space", xaxis_title="Component 1", yaxis_title="Component 2", height=500)
    counts = metadata["cluster_label"].value_counts().sort_index() if "cluster_label" in metadata.columns else pd.Series(dtype=int)
    secondary = html.P("Clusters are exploratory subgroups in the current delta-feature space.")
    secondary_figure = go.Figure()
    if not counts.empty:
        secondary_figure.add_trace(go.Bar(x=[str(index) for index in counts.index.tolist()], y=counts.tolist()))
    secondary_figure.update_layout(title="Cluster sizes", xaxis_title="Cluster", yaxis_title="n", height=360)
    return summary, figure, secondary, secondary_figure


def _build_patterns_feature_view(patterns_result: dict[str, Any]) -> tuple[Any, go.Figure, Any, go.Figure]:
    feature_patterns = patterns_result.get("feature_pattern_results", {})
    if feature_patterns.get("message"):
        placeholder = html.P(feature_patterns["message"])
        return placeholder, _empty_figure("Feature patterns"), placeholder, _empty_figure("Ordered pattern matrix")
    ordered_labels = feature_patterns.get("ordered_labels", [])
    ordered_corr = np.asarray(feature_patterns.get("ordered_correlation_matrix", []), dtype=float)
    summary = html.Div(
        [
            html.P(f"Level: {feature_patterns.get('level', 'n/a')}"),
            html.P(f"Pattern count: {len(ordered_labels)}"),
            html.P("Correlation-based hierarchical ordering reveals edges or ROI profiles that change together across subjects."),
        ]
    )
    figure = go.Figure(data=go.Heatmap(z=ordered_corr, x=ordered_labels, y=ordered_labels, colorscale="RdBu_r", zmin=-1.0, zmax=1.0, zmid=0.0))
    figure.update_layout(title="Ordered feature correlation matrix", height=520)
    secondary = html.P("Top ordered labels are shown below.")
    secondary_figure = go.Figure()
    if ordered_labels:
        secondary_figure.add_trace(go.Bar(x=ordered_labels[:20], y=list(range(1, min(len(ordered_labels), 20) + 1))))
    secondary_figure.update_layout(title="Ordered feature sequence", xaxis_title="Feature", yaxis_title="Order position", height=360)
    return summary, figure, secondary, secondary_figure


def _build_patterns_brain_behavior_view(patterns_result: dict[str, Any]) -> tuple[Any, go.Figure, Any, go.Figure]:
    brain_behavior = patterns_result.get("brain_behavior_results", {})
    if brain_behavior.get("message"):
        placeholder = html.P(brain_behavior["message"])
        return placeholder, _empty_figure("Brain-behavior"), placeholder, _empty_figure("Brain-behavior weights")
    scores = pd.DataFrame(brain_behavior.get("scores", []))
    summary = html.Div(
        [
            html.P(f"Method: {brain_behavior.get('method', 'n/a').upper()}"),
            html.P("Behavior variables: " + ", ".join(brain_behavior.get("behavior_variables", []))),
            html.P("Latent singular values: " + ", ".join(f"{value:.3f}" for value in brain_behavior.get("singular_values", []))),
            _build_patterns_brain_behavior_weights_table(brain_behavior),
        ]
    )
    figure = go.Figure()
    if {"brain_score_1", "behavior_score_1"}.issubset(scores.columns):
        figure.add_trace(
            go.Scatter(
                x=scores["brain_score_1"],
                y=scores["behavior_score_1"],
                mode="markers",
                text=_build_patterns_hover_labels(scores),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figure.update_layout(title="Brain score vs behavior score", xaxis_title="Brain latent score 1", yaxis_title="Behavior latent score 1", height=500)
    secondary = html.P("The second chart highlights the strongest connectivity weights of the first latent component.")
    secondary_figure = go.Figure()
    brain_weights = brain_behavior.get("brain_feature_weights", [])
    if brain_weights:
        first = brain_weights[0]["features"]
        secondary_figure.add_trace(go.Bar(x=[row["feature"] for row in first], y=[row["weight"] for row in first]))
    secondary_figure.update_layout(title="Top connectivity weights", xaxis_title="Feature", yaxis_title="Weight", height=420)
    return summary, figure, secondary, secondary_figure


def _build_patterns_hover_labels(metadata: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in metadata.iterrows():
        parts = []
        for column in ("idx", "subject_id", "base_subject_id", "group_label", "timepoint_label"):
            if column in metadata.columns and pd.notna(row.get(column)):
                parts.append(f"{column}={row[column]}")
        labels.append("<br>".join(parts) if parts else "Observation")
    return labels


def _coerce_color_values(values: pd.Series | list[Any]) -> list[Any]:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.tolist()
    return pd.factorize(series.astype(str))[0].tolist()


def _build_patterns_loading_table(frame: pd.DataFrame) -> html.Table:
    if frame.empty:
        return _build_simple_table([("Status", "No loadings available")])
    rows = []
    for _, row in frame.iterrows():
        features = row.get("features", [])
        text = ", ".join(f"{item['feature']} ({item['loading']:.3f})" for item in features[:5])
        rows.append((f"Component {int(row['component'])}", text))
    return _build_simple_table(rows)


def _build_patterns_cluster_summary_table(frame: pd.DataFrame) -> html.Table:
    if frame.empty:
        return _build_simple_table([("Status", "No cluster summary available")])
    return html.Table(
        [
            html.Thead(html.Tr([html.Th("Cluster", style=_stats_cell_style(header=True)), html.Th("n", style=_stats_cell_style(header=True)), html.Th("Groups", style=_stats_cell_style(header=True)), html.Th("Timepoints", style=_stats_cell_style(header=True))])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(str(int(row["cluster_label"])), style=_stats_cell_style(label=True)),
                            html.Td(str(int(row["n"])), style=_stats_cell_style()),
                            html.Td(str(row["groups"]), style=_stats_cell_style()),
                            html.Td(str(row["timepoints"]), style=_stats_cell_style()),
                        ]
                    )
                    for _, row in frame.iterrows()
                ]
            ),
        ],
        style=_stats_table_style(),
    )


def _build_patterns_brain_behavior_weights_table(brain_behavior: dict[str, Any]) -> html.Table:
    rows: list[tuple[str, str]] = []
    for component in brain_behavior.get("behavior_weights", [])[:2]:
        text = ", ".join(f"{item['variable']} ({item['weight']:.3f})" for item in component.get("variables", []))
        rows.append((f"Behavior component {component['component']}", text))
    for component in brain_behavior.get("brain_feature_weights", [])[:2]:
        text = ", ".join(f"{item['feature']} ({item['weight']:.3f})" for item in component.get("features", [])[:5])
        rows.append((f"Brain component {component['component']}", text))
    return _build_simple_table(rows if rows else [("Status", "No weights available")])


def _build_network_help_content() -> list[Any]:
    return [
        html.P(
            "The Network tab summarizes subject-level delta matrices (Trial B - Trial A) into compact ROI, network, graph, module, and NBS views."
        ),
        html.Div("Modes", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Summary: ROI-, subnet-, and laterality-based scores with group- or longitudinal inference."),
                html.Li("Graph: node-wise and global graph metrics computed from the current weighted delta matrices."),
                html.Li("Modules: exploratory component/community view on thresholded mean network structure."),
                html.Li("NBS: Network-Based Statistic style component inference using a primary edge threshold and permutations."),
            ]
        ),
        html.Div("Controls", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Score family / metric: selects which score family is highlighted in the plots and summary tables."),
                html.Li("Weight mode: defines how edge deltas are turned into graph weights."),
                html.Li("Threshold mode / value: optionally removes weak edges before graph, module, or NBS-style summaries."),
                html.Li("NBS primary threshold: edge-level test statistic threshold used to define suprathreshold components."),
                html.Li("NBS permutations: number of random relabelings/sign-flips used to estimate component-level p-values."),
            ]
        ),
        html.Div("Weighting And Thresholds", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Positive weights: negative deltas are clipped to zero before graph analysis."),
                html.Li("Absolute weights: graph analysis uses absolute delta magnitude regardless of sign."),
                html.Li("Raw positive part: keeps the current positive-only weighted interpretation for graph measures."),
                html.Li("None threshold: use all retained weighted edges."),
                html.Li("Absolute weight threshold: keep only edges with weight >= threshold value."),
                html.Li("Density threshold: keep only the strongest fraction of edges based on the chosen density."),
            ]
        ),
        html.Div("Summary Measures", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("ROI mean connectivity: mean delta from one ROI to all other ROIs."),
                html.Li("Ipsilateral / contralateral mean: average connectivity to ROIs in the same or opposite hemisphere."),
                html.Li("Within-class / between-class mean: average connectivity inside or outside the ROI's anatomical class."),
                html.Li("Within-network connectivity: average edge delta among ROIs of one anatomical subnet."),
                html.Li("Between-network connectivity: average edge delta between two anatomical subnet classes."),
                html.Li("Laterality index (LI): normalized left-right asymmetry, usually (left - right) / (|left| + |right|)."),
            ]
        ),
        html.Div("Graph Metrics", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Node strength: sum of incident edge weights at one ROI."),
                html.Li("Degree: number of suprathreshold edges incident on one ROI."),
                html.Li("Clustering coefficient: tendency of a node's neighbors to also be mutually connected."),
                html.Li("Local efficiency: communication efficiency inside the neighborhood of one ROI."),
                html.Li("Global efficiency: inverse-shortest-path based integration of the whole network."),
                html.Li("Betweenness: how often an ROI lies on shortest paths between other ROIs."),
                html.Li("Participation coefficient: how evenly a node connects across multiple components/modules."),
                html.Li("Hubness: composite ranking based on centrality-related node metrics."),
                html.Li("Modularity: strength of separation between components/modules in the weighted network."),
            ]
        ),
        html.Div("Modules And NBS", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Modules / communities in this version are exploratory connected components on the thresholded network, not optimized Louvain/Leiden partitions."),
                html.Li("NBS = Network-Based Statistic. Instead of correcting every edge independently, suprathreshold edges are grouped into connected components."),
                html.Li("Component p-values in NBS are permutation-based and refer to network components, not to single edges."),
            ]
        ),
        html.Div("Abbreviations", style=_stats_section_title_style()),
        _build_simple_table(
            [
                ("ROI", "Region of interest"),
                ("LI", "Laterality index"),
                ("NBS", "Network-Based Statistic"),
                ("n", "Number of observations contributing to the score"),
                ("p", "Raw p-value"),
                ("q", "Multiplicity-corrected p-value"),
                ("Delta", "Subject-level contrast Trial B - Trial A"),
            ]
        ),
    ]


def _build_patterns_help_content() -> list[Any]:
    return [
        html.P(
            "The Patterns tab explores subject-level edge-delta structure to reveal latent dimensions, subject subgroups, co-varying edges, and multivariate brain-behavior relationships."
        ),
        html.Div("Modes", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("Embedding: low-dimensional representation of subjects based on all edge deltas."),
                html.Li("Subject clusters: grouping of subjects based on whole-connectivity delta profiles."),
                html.Li("Feature patterns: clustering of edges or ROI profiles that vary together across subjects."),
                html.Li("CCA / PLS: multivariate relationship between connectivity patterns and numeric behavior variables."),
            ]
        ),
        html.Div("Current build", style=_stats_section_title_style()),
        html.Ul(
            [
                html.Li("PCA is fully available without extra dependencies."),
                html.Li("k-means and hierarchical clustering are available."),
                html.Li("PLS-style latent brain-behavior decomposition is available."),
                html.Li("t-SNE, UMAP, Gaussian mixtures, and direct CCA are shown in the UI but return an explanatory message if the needed dependencies are unavailable."),
            ]
        ),
        html.Div("Abbreviations", style=_stats_section_title_style()),
        _build_simple_table(
            [
                ("PCA", "Principal component analysis"),
                ("PLS", "Partial least squares"),
                ("CCA", "Canonical correlation analysis"),
                ("ROI", "Region of interest"),
                ("Delta", "Subject-level contrast Trial B - Trial A"),
            ]
        ),
    ]


def _build_summary_network_view(network_result: dict[str, Any], metric_name: str) -> tuple[Any, go.Figure, Any, go.Figure]:
    summary = network_result.get("network_summary_results", {})
    results = pd.DataFrame(summary.get("results", []))
    if results.empty:
        placeholder = html.P(summary.get("message", "No network summary results are available."))
        return placeholder, _empty_figure("Network summary"), placeholder, _empty_figure("Network detail")

    filtered = results[results["score_family"] == metric_name].copy()
    if filtered.empty:
        filtered = results.head(12).copy()
    filtered = filtered.sort_values("mean_delta", ascending=False)

    table = _build_network_results_table(filtered.head(10))
    figure = go.Figure(
        data=go.Bar(
            x=filtered["score_label"].tolist()[:12],
            y=filtered["mean_delta"].tolist()[:12],
            text=[_format_float(value, 4) for value in filtered["q_value"].tolist()[:12]],
            hovertemplate="Score=%{x}<br>Mean delta=%{y:.4f}<br>q=%{text}<extra></extra>",
        )
    )
    figure.update_layout(title="Network summary scores", xaxis_title="Score", yaxis_title="Mean delta", height=440)

    secondary = html.Div(
        [
            html.P("The table shows inferential summaries for the selected network score family."),
            table,
        ]
    )
    heatmap = go.Figure(
        data=go.Heatmap(
            z=np.array([filtered["mean_delta"].tolist()[:12]], dtype=float),
            x=filtered["score_label"].tolist()[:12],
            y=["Mean delta"],
            colorscale="RdBu_r",
            zmid=0.0,
            text=[[ _format_float(value, 4) for value in filtered["q_value"].tolist()[:12] ]],
            hovertemplate="Score=%{x}<br>Mean delta=%{z:.4f}<br>q=%{text}<extra></extra>",
        )
    )
    heatmap.update_layout(title="Summary score strip", height=260)
    return secondary, figure, html.P("Filtered summary overview"), heatmap


def _build_graph_network_view(network_result: dict[str, Any], metric_name: str) -> tuple[Any, go.Figure, Any, go.Figure]:
    graph_result = network_result.get("graph_metric_results", {})
    global_frame = pd.DataFrame(graph_result.get("global", {}).get("results", []))
    node_frame = pd.DataFrame(graph_result.get("node", {}).get("results", []))
    combined = pd.concat([global_frame, node_frame], ignore_index=True) if not global_frame.empty or not node_frame.empty else pd.DataFrame()
    if combined.empty:
        placeholder = html.P("No graph metrics are available for the current selection.")
        return placeholder, _empty_figure("Graph metrics"), placeholder, _empty_figure("Node ranking")

    filtered = combined[combined["score_family"] == metric_name].copy()
    if filtered.empty:
        filtered = combined.head(12).copy()
    filtered = filtered.sort_values("mean_delta", ascending=False)
    summary = html.Div([html.P("Graph metrics are computed on subject-level delta matrices after the selected weighting and threshold rules."), _build_network_results_table(filtered.head(10))])

    figure = go.Figure(
        data=go.Bar(
            x=filtered["score_label"].tolist()[:12],
            y=filtered["mean_delta"].tolist()[:12],
            marker_color="#1f77b4",
        )
    )
    figure.update_layout(title="Graph metric ranking", xaxis_title="Metric", yaxis_title="Mean delta", height=440)

    node_only = node_frame[node_frame["score_family"] == metric_name].copy()
    if node_only.empty:
        node_only = node_frame.head(12).copy()
    secondary = html.Div([html.P("Node-wise metrics highlight ROIs with the largest systematic changes."), _build_network_results_table(node_only.head(10)) if not node_only.empty else html.P("No node-wise metrics for this selection.")])
    secondary_figure = go.Figure()
    if not node_only.empty:
        secondary_figure.add_trace(go.Bar(x=node_only["score_label"].tolist()[:12], y=node_only["mean_delta"].tolist()[:12], marker_color="#ff7f0e"))
    secondary_figure.update_layout(title="Top ROI metrics", xaxis_title="ROI metric", yaxis_title="Mean delta", height=440)
    return summary, figure, secondary, secondary_figure


def _build_modules_network_view(network_result: dict[str, Any]) -> tuple[Any, go.Figure, Any, go.Figure]:
    community = network_result.get("community_results", {})
    communities = community.get("communities", {})
    if not communities:
        placeholder = html.P(community.get("message", "No module analysis is available."))
        return placeholder, _empty_figure("Community matrix"), placeholder, _empty_figure("Community assignments")

    first_label = next(iter(communities))
    first = communities[first_label]
    summary = html.Div(
        [
            html.P(community.get("note", "")),
            html.P(f"Showing component structure for: {first_label}"),
            _build_simple_table([("Component count", str(first.get("component_count", "n/a")))]),
        ]
    )
    matrix = np.asarray(first.get("matrix", []), dtype=float)
    figure = go.Figure(data=go.Heatmap(z=matrix, colorscale="RdBu_r", zmid=0.0))
    figure.update_layout(title=f"Community input matrix: {first_label}", height=480)

    component_frame = pd.DataFrame(first.get("components", []))
    secondary = html.Div([html.P("Components are exploratory connected suprathreshold modules."), _build_network_component_table(component_frame)])
    secondary_figure = go.Figure()
    if not component_frame.empty:
        secondary_figure.add_trace(go.Scatter(x=component_frame["roi"], y=component_frame["component"], mode="markers+text", text=component_frame["component"], textposition="top center"))
    secondary_figure.update_layout(title="ROI-to-component assignment", xaxis_title="ROI", yaxis_title="Component", height=360)
    return summary, figure, secondary, secondary_figure


def _build_nbs_network_view(network_result: dict[str, Any]) -> tuple[Any, go.Figure, Any, go.Figure]:
    nbs = network_result.get("nbs_results", {})
    if nbs.get("message"):
        placeholder = html.P(nbs["message"])
        return placeholder, _empty_figure("NBS"), placeholder, _empty_figure("NBS components")

    components = pd.DataFrame(nbs.get("components", []))
    edges = pd.DataFrame(nbs.get("suprathreshold_edges", []))
    summary = html.Div(
        [
            html.P(f"Primary threshold: {nbs.get('primary_threshold', 'n/a')}"),
            html.P(f"Permutations: {nbs.get('permutations', 'n/a')}"),
            html.P("NBS is computed separately from ROI-pair q-values and reports component-wise significance."),
            _build_network_nbs_table(components),
        ]
    )

    figure = _empty_figure("NBS suprathreshold edges")
    if not edges.empty:
        pivot = edges.pivot(index="roi_from", columns="roi_to", values="statistic").fillna(0.0)
        figure = go.Figure(data=go.Heatmap(z=pivot.to_numpy(dtype=float), x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale="RdBu_r", zmid=0.0))
        figure.update_layout(title="Suprathreshold edge statistics", height=480)

    secondary = html.Div([html.P("Each bar represents the size of one suprathreshold component."), _build_network_nbs_table(components)])
    secondary_figure = go.Figure()
    if not components.empty:
        secondary_figure.add_trace(go.Bar(x=[f"Component {index + 1}" for index in range(len(components))], y=components["edge_count"].tolist(), text=[_format_float(value, 4) for value in components["p_value"]], hovertemplate="%{x}<br>Edges=%{y}<br>p=%{text}<extra></extra>"))
    secondary_figure.update_layout(title="NBS component sizes", xaxis_title="Component", yaxis_title="Edge count", height=360)
    return summary, figure, secondary, secondary_figure


def _build_network_results_table(frame: pd.DataFrame) -> html.Table:
    if frame.empty:
        return _build_simple_table([("Status", "No results")])
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Score", style=_stats_cell_style(header=True)),
                        html.Th("Mean delta", style=_stats_cell_style(header=True)),
                        html.Th("Effect", style=_stats_cell_style(header=True)),
                        html.Th("p", style=_stats_cell_style(header=True)),
                        html.Th("q", style=_stats_cell_style(header=True)),
                        html.Th("n", style=_stats_cell_style(header=True)),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(str(row["score_label"]), style=_stats_cell_style(label=True)),
                            html.Td(_format_float(row.get("mean_delta"), 4), style=_stats_cell_style()),
                            html.Td(_format_float(row.get("effect_size"), 4), style=_stats_cell_style()),
                            html.Td(_format_float(row.get("p_value"), 4), style=_stats_cell_style()),
                            html.Td(_format_float(row.get("q_value"), 4), style=_stats_cell_style()),
                            html.Td(str(int(row.get("n", 0))), style=_stats_cell_style()),
                        ]
                    )
                    for _, row in frame.iterrows()
                ]
            ),
        ],
        style=_stats_table_style(),
    )


def _build_network_component_table(frame: pd.DataFrame) -> html.Table:
    if frame.empty:
        return _build_simple_table([("Status", "No components")])
    return html.Table(
        [
            html.Thead(html.Tr([html.Th("ROI", style=_stats_cell_style(header=True)), html.Th("Component", style=_stats_cell_style(header=True))])),
            html.Tbody([html.Tr([html.Td(str(row["roi"]), style=_stats_cell_style(label=True)), html.Td(str(int(row["component"])), style=_stats_cell_style())]) for _, row in frame.iterrows()]),
        ],
        style=_stats_table_style(),
    )


def _build_network_nbs_table(frame: pd.DataFrame) -> html.Table:
    if frame.empty:
        return _build_simple_table([("Status", "No suprathreshold components")])
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Component", style=_stats_cell_style(header=True)),
                        html.Th("Edges", style=_stats_cell_style(header=True)),
                        html.Th("Nodes", style=_stats_cell_style(header=True)),
                        html.Th("p", style=_stats_cell_style(header=True)),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(f"Component {index + 1}", style=_stats_cell_style(label=True)),
                            html.Td(str(int(row.get("edge_count", 0))), style=_stats_cell_style()),
                            html.Td(", ".join(row.get("nodes", [])), style=_stats_cell_style()),
                            html.Td(_format_float(row.get("p_value"), 4), style=_stats_cell_style()),
                        ]
                    )
                    for index, (_, row) in enumerate(frame.iterrows())
                ]
            ),
        ],
        style=_stats_table_style(),
    )
