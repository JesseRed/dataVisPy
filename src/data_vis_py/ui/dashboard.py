"""Dash application factory for the connectivity analysis dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from data_vis_py.io.dataset_loader import DatasetBundle, load_dataset
from data_vis_py.stats.analysis import AnalysisConfig, run_analysis, run_covariate_analysis, summarize_pair_result


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
        Output("regression-covariates", "options"),
        Output("regression-covariates", "value"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
    )
    def sync_file_dependent_controls(json_filename: str, csv_filename: str):
        bundle = current_bundle(json_filename, csv_filename)
        groups = sorted(bundle.subjects["group_label"].dropna().astype(str).unique().tolist())
        group_options = [{"label": group, "value": group} for group in groups]
        numeric_columns = [
            column
            for column in bundle.subjects.columns
            if column not in {"dataset_id", "subject_id", "base_subject_id", "group_label", "mtime"}
            and str(bundle.subjects[column].dtype) != "object"
        ]
        numeric_options = [{"label": column, "value": column} for column in numeric_columns]
        default_group = groups[0] if groups else None
        default_corr = numeric_columns[0] if numeric_columns else None
        default_regression = numeric_columns[:2]
        return group_options, default_group, group_options, default_group, numeric_options, default_corr, numeric_options, default_regression

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
        Output("pair-summary", "children"),
        Output("pair-detail-chart", "figure"),
        Output("covariate-summary", "children"),
        Output("covariate-chart", "figure"),
        Input("json-file", "value"),
        Input("csv-file", "value"),
        Input("analysis-mode", "value"),
        Input("group-a", "value"),
        Input("group-b", "value"),
        Input("longitudinal-enabled", "value"),
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
        Input("correlation-method", "value"),
        Input("correlation-variable", "value"),
        Input("regression-covariates", "value"),
        Input("selected-pair", "data"),
    )
    def update_dashboard(
        json_filename: str,
        csv_filename: str,
        analysis_mode: str,
        group_a: str,
        group_b: str,
        longitudinal_enabled_flags: list[str],
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
        correlation_method: str,
        correlation_variable: str | None,
        regression_covariates: list[str] | None,
        selected_pair: str,
    ) -> tuple[Any, Any, go.Figure, go.Figure, Any, Any, go.Figure, Any, go.Figure]:
        bundle = current_bundle(json_filename, csv_filename)
        longitudinal_enabled = "enabled" in (longitudinal_enabled_flags or [])
        if BAND_PRESETS.get(band_preset):
            freq_min, freq_max = BAND_PRESETS[band_preset]
        freq_min = float(freq_min)
        freq_max = float(freq_max)
        if freq_min > freq_max:
            freq_min, freq_max = freq_max, freq_min
        config = AnalysisConfig(
            analysis_mode=analysis_mode,
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
            longitudinal_column="mtime",
            longitudinal_value_a=longitudinal_value_a,
            longitudinal_value_b=longitudinal_value_b,
        )
        analysis_result = run_analysis(bundle, config)
        effective_analysis_mode = "longitudinal_delta" if longitudinal_enabled else analysis_mode
        pair_summary = summarize_pair_result(analysis_result, selected_pair)
        covariate_result = run_covariate_analysis(
            bundle,
            analysis_result,
            selected_pair,
            correlation_method=correlation_method,
            correlation_variable=correlation_variable,
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
                    f"Longitudinal mode: MTime {longitudinal_value_a} vs {longitudinal_value_b}"
                    if longitudinal_enabled
                    else "Longitudinal mode: off"
                ),
            ]
        )

        heatmap_description = html.Div(
            [
                html.H4("Current heatmap calculation"),
                html.P(
                    f"{analysis_result['analysis_label']} with group selection "
                    f"{analysis_result['selected_group_a']}"
                    f"{' vs ' + str(analysis_result['selected_group_b']) if analysis_result['between_groups'] else ''} "
                    f"over {analysis_result['freq_label']} using {analysis_result['test_label']}."
                ),
                html.P(
                    f"Heatmap mode: {DISPLAY_OPTIONS[display_mode][0]}. "
                    f"Multiple-testing correction: {analysis_result['correction_label']}."
                ),
                html.P(
                    "In p-value and q-value view, cells outlined in red are below 0.05 under the currently selected correction rule."
                ),
            ]
        )

        heatmap_figure = _build_heatmap(
            analysis_result=analysis_result,
            display_mode=display_mode,
            selected_pair=selected_pair,
        )
        subject_bar_figure = _build_subject_bar_chart(pair_summary, effective_analysis_mode)
        subject_bar_stats = _build_subject_bar_stats(pair_summary)

        pair_component, pair_figure = _build_detail_view(pair_summary, effective_analysis_mode)
        cov_component, cov_figure = _build_covariate_view(covariate_result, effective_analysis_mode, correlation_variable)
        return summary_component, heatmap_description, heatmap_figure, subject_bar_figure, subject_bar_stats, pair_component, pair_figure, cov_component, cov_figure

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
            html.Label("Analysis Mode"),
            dcc.RadioItems(
                id="analysis-mode",
                options=[
                    {"label": "Trial Contrast", "value": "trial_delta"},
                    {"label": "Session Contrast", "value": "session_delta"},
                ],
                value="trial_delta",
            ),
            html.Label("Group 1"),
            dcc.Dropdown(
                id="group-a",
                options=[{"label": group, "value": group} for group in group_options],
                value=group_options[0] if group_options else None,
                clearable=False,
            ),
            html.Label("Group 2"),
            dcc.Dropdown(
                id="group-b",
                options=[{"label": group, "value": group} for group in group_options],
                value=group_options[0] if group_options else None,
                clearable=False,
            ),
            html.Label("Longitudinal analysis"),
            dcc.Checklist(
                id="longitudinal-enabled",
                options=[{"label": "Enable longitudinal delta-vs-delta analysis", "value": "enabled"}],
                value=[],
            ),
            html.Label("Timepoint 1"),
            dcc.Dropdown(id="longitudinal-value-a", options=[], value=None, clearable=False),
            html.Label("Timepoint 2"),
            dcc.Dropdown(id="longitudinal-value-b", options=[], value=None, clearable=False),
            html.Label("Trial A / fixed trial"),
            dcc.Dropdown(id="trial-a", options=trial_options, value=bundle.trial_ids[0], clearable=False),
            html.Label("Trial B"),
            dcc.Dropdown(id="trial-b", options=trial_options, value=bundle.trial_ids[1], clearable=False),
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
                value="effect",
            ),
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
            html.Label("Regression covariates"),
            dcc.Dropdown(
                id="regression-covariates",
                options=[{"label": column, "value": column} for column in numeric_subject_columns],
                value=numeric_subject_columns[:2],
                multi=True,
            ),
        ],
        style={"display": "grid", "gap": "8px"},
    )


def _build_heatmap(analysis_result: dict[str, Any], display_mode: str, selected_pair: str) -> go.Figure:
    display_label, key = DISPLAY_OPTIONS[display_mode]
    z_values = np.array(analysis_result[key], dtype=float)
    pair_text = np.empty_like(z_values, dtype=object)
    value_text = np.empty_like(z_values, dtype=object)
    for i, roi_from in enumerate(analysis_result["roi_order"]):
        for j, roi_to in enumerate(analysis_result["roi_order"]):
            pair_text[i, j] = f"{roi_from} -> {roi_to}"
            value = z_values[i, j]
            value_text[i, j] = "" if np.isnan(value) else f"{value:.3f}"

    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=analysis_result["roi_order"],
            y=analysis_result["roi_order"],
            colorscale="RdBu_r" if display_mode == "effect" else "Viridis",
            zmid=0 if display_mode == "effect" else None,
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
                if i == j or np.isnan(significance[i, j]) or significance[i, j] >= 0.05:
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


def _build_subject_bar_chart(pair_summary: dict[str, Any] | None, analysis_mode: str) -> go.Figure:
    figure = go.Figure()
    if not pair_summary:
        figure.update_layout(title="Subject values for selected edge")
        return figure

    detail_frame = pd.DataFrame(pair_summary["detail_records"])
    if detail_frame.empty:
        figure.update_layout(title="Subject values for selected edge")
        return figure

    if analysis_mode == "trial_delta":
        id_column = "subject_id"
        y_column = "delta"
        chart_title = f"Subject deltas for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
        yaxis_title = "Trial B - Trial A"
    elif analysis_mode == "session_delta":
        id_column = "base_subject_id"
        y_column = "delta"
        chart_title = f"Session deltas for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
        yaxis_title = "M2 - M1"
    else:
        id_column = "base_subject_id"
        y_column = "delta"
        chart_title = f"Longitudinal delta differences for {pair_summary['roi_from']} -> {pair_summary['roi_to']}"
        yaxis_title = "Delta(B) - Delta(A)"

    if "group_label" not in detail_frame.columns:
        detail_frame["group_label"] = "All"
    detail_frame["group_label"] = detail_frame["group_label"].astype(str)

    group_order = [entry["group_label"] for entry in pair_summary.get("group_stats", [])]
    if not group_order:
        group_order = list(dict.fromkeys(detail_frame["group_label"].tolist()))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    color_map = {group: palette[index % len(palette)] for index, group in enumerate(group_order)}

    category_array: list[str] = []
    has_multiple_groups = len(group_order) > 1
    for index, group_name in enumerate(group_order):
        group_frame = detail_frame[detail_frame["group_label"] == group_name].copy()
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
                        group_frame["group_label"],
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
            "title": "IDX" if analysis_mode != "trial_delta" else "Subject",
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


def _stats_table_style() -> dict[str, str]:
    return {
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "13px",
        "backgroundColor": "#ffffff",
        "border": "1px solid #d9dde3",
    }


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
        if analysis_mode == "trial_delta":
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
        elif analysis_mode == "session_delta":
            figure.add_trace(
                go.Scatter(
                    x=detail_frame["m1_value"],
                    y=detail_frame["m2_value"],
                    mode="markers+text",
                    text=detail_frame["base_subject_id"],
                    textposition="top center",
                    name="Subjects",
                )
            )
            figure.update_layout(
                title="M1 vs M2",
                xaxis_title="M1 value",
                yaxis_title="M2 value",
                height=420,
            )
        else:
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
    return summary, figure


def _build_covariate_view(
    covariate_result: dict[str, Any],
    analysis_mode: str,
    correlation_variable: str | None,
) -> tuple[Any, go.Figure]:
    if covariate_result.get("message"):
        return html.P(covariate_result["message"]), go.Figure()

    lines = [html.H4("Covariate analysis")]
    if analysis_mode in {"session_delta", "longitudinal_delta"}:
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
            )
        )
        figure.update_layout(
            title=f"Delta vs {correlation_variable}",
            xaxis_title=correlation_variable,
            yaxis_title="Connectivity delta",
            height=420,
        )

    return html.Div(lines), figure
