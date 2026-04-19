"""Connectivity analysis helpers used by the Dash UI and tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

from data_vis_py.io.dataset_loader import DatasetBundle


CORRECTION_METHODS = ("none", "fdr_bh", "bonferroni", "holm")
ALL_GROUPS_VALUE = "__all_groups__"
ALL_GROUPS_LABEL = "All groups"
SELECTED_EDGE_DELTA_VALUE = "__selected_edge_delta__"
SELECTED_EDGE_DELTA_LABEL = "Selected-edge delta"


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for a single analysis request."""

    dataset_id: str
    metric: str
    trial_a: int
    trial_b: int
    freq_min: float
    freq_max: float
    mtime_filter: str = "All"
    correction_method: str = "fdr_bh"
    group_a: str | None = None
    group_b: str | None = None
    longitudinal_enabled: bool = False
    longitudinal_require_pairs: bool = True
    longitudinal_column: str | None = None
    longitudinal_value_a: Any = None
    longitudinal_value_b: Any = None
    excluded_idx: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkAnalysisConfig:
    """Settings for network-level analyses derived from subject-level edge deltas."""

    mode: str = "summary"
    metric_name: str = "roi_mean_connectivity"
    weight_mode: str = "positive"
    threshold_mode: str = "none"
    threshold_value: float = 0.0
    network_labeling_mode: str = "anatomical"
    nbs_primary_threshold: float = 2.0
    nbs_permutations: int = 200


@dataclass(frozen=True)
class PatternsAnalysisConfig:
    """Settings for exploratory pattern analysis on subject-level edge-delta features."""

    mode: str = "embedding"
    feature_basis: str = "edge_deltas"
    standardize: str = "zscore"
    missing_data_mode: str = "complete_cases"
    embedding_method: str = "pca"
    embedding_dim: int = 2
    color_variable: str = "group_label"
    cluster_method: str = "kmeans"
    n_clusters: int = 3
    feature_pattern_level: str = "edges"
    cca_pls_method: str = "pls"
    behavior_variables: tuple[str, ...] = ()
    n_components: int = 2


@dataclass(frozen=True)
class LongitudinalDetailConfig:
    """Settings for selected-edge longitudinal detail analyses."""

    model_family: str = "mixed_effects"
    random_slope_time: bool = False
    trajectory_ci: float = 0.95
    reliable_change_method: str = "jt_baseline_sd"
    baseline_value: Any = None
    followup_value: Any = None
    reliability_coefficient: float = 0.8


def run_analysis(bundle: DatasetBundle, config: AnalysisConfig) -> dict[str, Any]:
    """Run an analysis and return matrix and detail records."""
    filtered = bundle.connectivity[
        (bundle.connectivity["dataset_id"] == config.dataset_id)
        & (bundle.connectivity["metric"] == config.metric)
        & (bundle.connectivity["freq"] >= config.freq_min)
        & (bundle.connectivity["freq"] <= config.freq_max)
    ].copy()
    excluded_idx = set(_normalize_excluded_idx(config.excluded_idx))
    if excluded_idx and "idx" in filtered.columns:
        filtered = filtered[~filtered["idx"].astype(str).isin(excluded_idx)].copy()
    filtered = (
        filtered.groupby(
            ["dataset_id", "metric", "subject_id", "idx", "base_subject_id", "group_label", "mtime", "trial_id", "roi_from", "roi_to"],
            as_index=False,
        )["value"]
        .mean()
    )

    if config.longitudinal_enabled and config.longitudinal_column:
        if config.longitudinal_column not in filtered.columns:
            filtered = filtered.merge(
                bundle.subjects[["subject_id", "idx", config.longitudinal_column]],
                on=["subject_id", "idx"],
                how="left",
                validate="many_to_one",
            )

    if config.longitudinal_enabled and config.longitudinal_column:
        results, group_context = _longitudinal_trial_delta_results(filtered, config)
        analysis_label = (
            f"Longitudinal delta contrast: Trial {config.trial_b} - Trial {config.trial_a} | "
            f"{config.longitudinal_column} {config.longitudinal_value_b} - {config.longitudinal_value_a}"
        )
        if config.longitudinal_require_pairs:
            test_label = "Paired t-test on within-subject trial deltas" if not group_context["between_groups"] else "Welch t-test between groups on longitudinal deltas"
        else:
            test_label = "Welch t-test on trial deltas between timepoints" if not group_context["between_groups"] else "OLS interaction test on trial deltas (group x timepoint)"
    else:
        if config.mtime_filter != "All":
            filtered = filtered[filtered["mtime"] == config.mtime_filter]
        results, group_context = _trial_delta_results(filtered, config)
        analysis_label = f"Trial {config.trial_b} - Trial {config.trial_a}"
        test_label = "Paired t-test" if not group_context["between_groups"] else "Welch t-test between groups on trial deltas"

    roi_order = list(bundle.channels)
    matrix = _build_matrix(results, roi_order, "mean_delta")
    p_matrix = _build_matrix(results, roi_order, "p_value")
    q_matrix = _build_matrix(results, roi_order, "q_value")
    significance_matrix = _build_matrix(
        results,
        roi_order,
        "p_value" if config.correction_method == "none" else "q_value",
    )

    pair_records = results.sort_values(["roi_from", "roi_to"]).to_dict(orient="records")
    details = {record["pair_key"]: record["detail_records"] for record in pair_records}
    groups = sorted(str(value) for value in filtered["group_label"].dropna().unique())
    group_label = ", ".join(groups) if groups else "Unknown"

    return {
        "analysis_label": analysis_label,
        "dataset_id": config.dataset_id,
        "metric": config.metric,
        "trial_a": config.trial_a,
        "trial_b": config.trial_b,
        "freq_label": f"{config.freq_min:.3f} - {config.freq_max:.3f} Hz",
        "freq_min": config.freq_min,
        "freq_max": config.freq_max,
        "test_label": test_label,
        "correction_method": config.correction_method,
        "correction_label": _correction_label(config.correction_method),
        "group_label": group_label,
        "selected_group_a": group_context["group_a"],
        "selected_group_b": group_context["group_b"],
        "between_groups": group_context["between_groups"],
        "longitudinal_enabled": config.longitudinal_enabled,
        "longitudinal_require_pairs": config.longitudinal_require_pairs,
        "longitudinal_column": config.longitudinal_column,
        "longitudinal_value_a": config.longitudinal_value_a,
        "longitudinal_value_b": config.longitudinal_value_b,
        "excluded_idx": tuple(sorted(excluded_idx)),
        "roi_order": roi_order,
        "matrix": matrix.tolist(),
        "p_matrix": p_matrix.tolist(),
        "q_matrix": q_matrix.tolist(),
        "significance_matrix": significance_matrix.tolist(),
        "pair_results": pair_records,
        "details": details,
    }


def summarize_pair_result(analysis_result: dict[str, Any], pair_key: str) -> dict[str, Any] | None:
    """Return the summary record for a selected pair."""
    for record in analysis_result["pair_results"]:
        if record["pair_key"] == pair_key:
            return record
    return None


def run_covariate_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    pair_key: str,
    *,
    correlation_method: str,
    correlation_variable: str | None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Compute correlation and regression outputs for the selected edge."""
    detail_records = analysis_result["details"].get(pair_key, [])
    if not detail_records:
        return {"message": "No edge is selected yet."}

    detail_frame = pd.DataFrame(detail_records)
    if detail_frame.empty:
        return {"message": "No observations are available for this edge."}

    if "subject_id" in detail_frame.columns:
        join_columns = ["subject_id"]
        subject_columns = [column for column in bundle.subjects.columns if column not in detail_frame.columns or column == "subject_id"]
        merged = detail_frame.merge(bundle.subjects[subject_columns], on="subject_id", how="left")
    else:
        baseline = (
            bundle.subjects.assign(mtime_order=bundle.subjects["mtime"].str.extract(r"(\d+)").fillna("999").astype(int))
            .sort_values(["base_subject_id", "mtime_order"])
            .drop_duplicates("base_subject_id", keep="first")
            .drop(columns="mtime_order")
        )
        join_columns = ["base_subject_id"]
        baseline_columns = [column for column in baseline.columns if column not in detail_frame.columns or column == "base_subject_id"]
        merged = detail_frame.merge(baseline[baseline_columns], on="base_subject_id", how="left")

    if "group_label" in detail_frame.columns:
        merged["group_label"] = detail_frame["group_label"].astype(str)
    elif "group_label" in merged.columns:
        merged["group_label"] = merged["group_label"].astype(str)
    excluded_idx = set(_normalize_excluded_idx(analysis_result.get("excluded_idx", ())))
    if excluded_idx and "idx" in merged.columns:
        merged = merged[~merged["idx"].astype(str).isin(excluded_idx)].copy()

    result: dict[str, Any] = {"joined_columns": join_columns, "message": None}

    if correlation_variable:
        correlation_columns = ["delta", correlation_variable]
        if "group_label" in merged.columns:
            correlation_columns.append("group_label")
        if "idx" in merged.columns:
            correlation_columns.append("idx")
        if "mtime" in merged.columns:
            correlation_columns.append("mtime")
        corr_frame = merged[correlation_columns].dropna().copy()
        if len(corr_frame) >= 3:
            result["correlation"] = _compute_correlation_summary(
                corr_frame,
                correlation_variable=correlation_variable,
                correlation_method=correlation_method,
            )
            if "group_label" in corr_frame.columns:
                group_results = []
                for group_name, group_frame in corr_frame.groupby("group_label", sort=False):
                    if len(group_frame) < 3:
                        group_results.append(
                            {
                                "group_label": str(group_name),
                                "variable": correlation_variable,
                                "method": correlation_method,
                                "n": int(len(group_frame)),
                                "message": "Not enough complete observations for correlation.",
                            }
                        )
                        continue
                    group_results.append(
                        _compute_correlation_summary(
                            group_frame,
                            correlation_variable=correlation_variable,
                            correlation_method=correlation_method,
                            group_label=str(group_name),
                        )
                    )
                result["correlation_by_group"] = group_results
                result["correlation_group_comparison"] = _compare_group_correlations(group_results, correlation_method)
        else:
            result["correlation"] = {
                "variable": correlation_variable,
                "method": correlation_method,
                "n": int(len(corr_frame)),
                "message": "Not enough complete observations for correlation.",
            }
            if "group_label" in corr_frame.columns:
                result["correlation_by_group"] = [
                    {
                        "group_label": str(group_name),
                        "variable": correlation_variable,
                        "method": correlation_method,
                        "n": int(len(group_frame)),
                        "message": "Not enough complete observations for correlation.",
                    }
                    for group_name, group_frame in corr_frame.groupby("group_label", sort=False)
                ]
                result["correlation_group_comparison"] = {
                    "method": correlation_method,
                    "message": "Not enough complete observations to compare group correlations.",
                }

    covariates = regression_covariates or []
    if covariates:
        regression_frame = merged.copy()
        if SELECTED_EDGE_DELTA_VALUE in covariates:
            regression_frame[SELECTED_EDGE_DELTA_VALUE] = regression_frame["delta"].astype(float)
        regression_frame = regression_frame[["delta"] + covariates].dropna()
        if len(regression_frame) >= max(4, len(covariates) + 1):
            predictors = sm.add_constant(regression_frame[covariates], has_constant="add")
            model = sm.OLS(regression_frame["delta"], predictors).fit()
            result["regression"] = {
                "variables": covariates,
                "n": int(len(regression_frame)),
                "r_squared": float(model.rsquared),
                "coefficients": {
                    name: {
                        "coef": float(model.params[name]),
                        "p_value": float(model.pvalues[name]),
                    }
                    for name in model.params.index
                },
            }
        else:
            result["regression"] = {
                "variables": covariates,
                "n": int(len(regression_frame)),
                "message": "Not enough complete observations for regression.",
            }

    return result


def run_multivariate_regression_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    pair_key: str,
    *,
    outcome_variable: str | None = None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a design-aware regression model for the selected edge."""
    detail_records = analysis_result["details"].get(pair_key, [])
    if not detail_records:
        return {"message": "No edge is selected yet."}

    detail_frame = pd.DataFrame(detail_records)
    if detail_frame.empty:
        return {"message": "No observations are available for this edge."}

    prepared = _prepare_regression_frame(bundle, analysis_result, detail_frame)
    model_frame = prepared["frame"]
    if model_frame.empty:
        return {"message": "No observations are available for regression after joining covariates."}

    outcome_variable = outcome_variable or SELECTED_EDGE_DELTA_VALUE
    covariates = regression_covariates or []
    if outcome_variable == SELECTED_EDGE_DELTA_VALUE:
        response_column = "selected_edge_delta"
        response_definition = prepared["default_response_definition"]
    else:
        if outcome_variable not in model_frame.columns:
            return {"message": f"The selected outcome variable '{outcome_variable}' is not available for this regression."}
        response_column = outcome_variable
        response_definition = outcome_variable

    resolved_covariates = []
    for covariate in covariates:
        if covariate == outcome_variable:
            continue
        resolved_covariates.append(covariate)
    covariates = resolved_covariates

    design_columns = list(prepared["design_columns"])
    if SELECTED_EDGE_DELTA_VALUE in covariates:
        model_frame[SELECTED_EDGE_DELTA_VALUE] = model_frame["selected_edge_delta"].astype(float)
    required_columns = [response_column] + design_columns + covariates
    regression_frame = model_frame.dropna(subset=required_columns).copy()

    if not design_columns and not covariates:
        return {
            "message": "Select at least one regression covariate or use a selection with an explicit group/timepoint design term.",
            "variables": covariates,
            "n": int(len(regression_frame)),
        }

    predictor_columns = design_columns + covariates
    min_required_rows = len(predictor_columns) + 2
    if len(regression_frame) < min_required_rows:
        return {
            "message": "Not enough complete observations for regression.",
            "variables": covariates,
            "n": int(len(regression_frame)),
            "required_n": int(min_required_rows),
        }

    predictors = sm.add_constant(regression_frame[predictor_columns], has_constant="add")
    model = sm.OLS(regression_frame[response_column], predictors).fit()
    confidence = model.conf_int()

    coefficient_rows = []
    for name in model.params.index:
        coefficient_rows.append(
            {
                "name": name,
                "label": prepared["coefficient_labels"].get(name, name),
                "estimate": float(model.params[name]),
                "std_error": float(model.bse[name]),
                "t_value": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
                "ci_low": float(confidence.loc[name, 0]),
                "ci_high": float(confidence.loc[name, 1]),
            }
        )

    point_labels = _build_regression_point_labels(regression_frame)
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag.astype(float)
    cooks_distance = np.asarray(influence.cooks_distance[0], dtype=float)
    studentized_residuals = np.asarray(influence.resid_studentized_external, dtype=float)
    diagnostics: dict[str, Any] = {
        "observed": regression_frame[response_column].astype(float).tolist(),
        "fitted": model.fittedvalues.astype(float).tolist(),
        "residuals": model.resid.astype(float).tolist(),
        "leverage": leverage.tolist(),
        "cooks_distance": cooks_distance.tolist(),
        "studentized_residuals": studentized_residuals.tolist(),
        "point_labels": point_labels,
        "observation_rows": _build_regression_observation_rows(
            regression_frame=regression_frame,
            point_labels=point_labels,
            observed=regression_frame[response_column].astype(float).to_numpy(dtype=float),
            fitted=model.fittedvalues.astype(float).to_numpy(dtype=float),
            residuals=model.resid.astype(float).to_numpy(dtype=float),
            leverage=leverage,
            cooks_distance=cooks_distance,
            studentized_residuals=studentized_residuals,
        ),
        "message": None,
    }
    if len(regression_frame) < max(6, len(predictor_columns) + 3):
        diagnostics["message"] = "Diagnostics are shown, but interpretation is limited because the regression sample is small."

    primary_effect_name = prepared.get("primary_effect_name")
    primary_effect = None
    if primary_effect_name and primary_effect_name in model.params.index:
        primary_effect = {
            "name": primary_effect_name,
            "label": prepared["coefficient_labels"].get(primary_effect_name, primary_effect_name),
            "estimate": float(model.params[primary_effect_name]),
            "std_error": float(model.bse[primary_effect_name]),
            "t_value": float(model.tvalues[primary_effect_name]),
            "p_value": float(model.pvalues[primary_effect_name]),
            "ci_low": float(confidence.loc[primary_effect_name, 0]),
            "ci_high": float(confidence.loc[primary_effect_name, 1]),
            "interpretation": prepared.get("primary_effect_interpretation"),
        }

    return {
        "message": None,
        "model_label": prepared["model_label"],
        "formula": _build_formula(response_definition, design_columns, covariates),
        "response_definition": response_definition,
        "observation_unit": prepared["observation_unit"],
        "design_effect_name": primary_effect_name,
        "design_effect_label": prepared.get("primary_effect_label"),
        "design_effect_interpretation": prepared.get("primary_effect_interpretation"),
        "methods_description": prepared["methods_description"],
        "n": int(len(regression_frame)),
        "variables": covariates,
        "coefficients": coefficient_rows,
        "primary_effect": primary_effect,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "diagnostics": diagnostics,
    }


def run_regression_influence_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    pair_key: str,
    *,
    outcome_variable: str | None = None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Return regression diagnostics focused on influence for the selected edge."""
    regression_result = run_multivariate_regression_analysis(
        bundle,
        analysis_result,
        pair_key,
        outcome_variable=outcome_variable,
        regression_covariates=regression_covariates,
    )
    if regression_result.get("message"):
        return regression_result
    diagnostics = regression_result.get("diagnostics", {})
    return {
        "message": None,
        "model_label": regression_result["model_label"],
        "formula": regression_result["formula"],
        "response_definition": regression_result["response_definition"],
        "n": regression_result["n"],
        "primary_effect": regression_result.get("primary_effect"),
        "diagnostics": diagnostics,
    }


def run_longitudinal_detail_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    pair_key: str,
    *,
    config: LongitudinalDetailConfig | None = None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Run selected-edge longitudinal detail analyses for the Heatmap panel."""
    config = config or LongitudinalDetailConfig()
    if not analysis_result.get("longitudinal_enabled"):
        return {"message": "Enable longitudinal mode to access mixed-effects, trajectory, and reliable-change analyses."}

    edge_frame = _build_edge_longitudinal_frame(bundle, analysis_result, pair_key)
    if edge_frame.empty:
        return {"message": "No longitudinal observations are available for the selected edge under the current filters."}

    selected_covariates = [
        column
        for column in (regression_covariates or [])
        if column != SELECTED_EDGE_DELTA_VALUE and column in edge_frame.columns and pd.api.types.is_numeric_dtype(edge_frame[column])
    ]
    result: dict[str, Any] = {
        "message": None,
        "pair_key": pair_key,
        "edge_frame_n": int(len(edge_frame)),
        "timepoints": edge_frame["timepoint_label"].astype(str).tolist(),
        "available_timepoints": edge_frame.sort_values("timepoint_order")["timepoint_label"].drop_duplicates().astype(str).tolist(),
        "covariates": selected_covariates,
    }

    if config.model_family == "change_score":
        result["primary_model"] = run_change_score_analysis(
            edge_frame,
            analysis_result,
            regression_covariates=selected_covariates,
            baseline_value=config.baseline_value,
            followup_value=config.followup_value,
        )
    elif config.model_family == "ancova":
        result["primary_model"] = run_ancova_analysis(
            edge_frame,
            analysis_result,
            regression_covariates=selected_covariates,
            baseline_value=config.baseline_value,
            followup_value=config.followup_value,
        )
    else:
        result["primary_model"] = run_mixed_effects_edge_analysis(
            edge_frame,
            analysis_result,
            regression_covariates=selected_covariates,
            random_slope_time=config.random_slope_time,
        )

    result["change_score"] = run_change_score_analysis(
        edge_frame,
        analysis_result,
        regression_covariates=selected_covariates,
        baseline_value=config.baseline_value,
        followup_value=config.followup_value,
    )
    result["ancova"] = run_ancova_analysis(
        edge_frame,
        analysis_result,
        regression_covariates=selected_covariates,
        baseline_value=config.baseline_value,
        followup_value=config.followup_value,
    )
    result["trajectory"] = run_edge_trajectory_analysis(edge_frame, analysis_result, ci_level=config.trajectory_ci)
    result["reliable_change"] = run_reliable_change_analysis(
        edge_frame,
        analysis_result,
        baseline_value=config.baseline_value,
        followup_value=config.followup_value,
        method=config.reliable_change_method,
        reliability_coefficient=config.reliability_coefficient,
    )
    return result


def run_mixed_effects_edge_analysis(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    regression_covariates: list[str] | None = None,
    random_slope_time: bool = False,
) -> dict[str, Any]:
    """Fit a selected-edge mixed-effects model on timepoint-level trial deltas."""
    frame = _selected_timepoint_frame(edge_frame, analysis_result)
    if frame.empty:
        return {"message": "No observations are available for the selected longitudinal comparison."}

    covariates = [column for column in (regression_covariates or []) if column in frame.columns]
    between_groups = bool(analysis_result.get("between_groups"))
    frame = frame.copy()
    frame["timepoint_indicator"] = (
        frame["timepoint_label"].astype(str) == str(analysis_result["longitudinal_value_b"])
    ).astype(float)
    predictor_columns = ["timepoint_indicator"]
    coefficient_labels = {
        "const": "Intercept",
        "timepoint_indicator": (
            f"Timepoint effect ({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})"
        ),
    }
    primary_effect_name = "timepoint_indicator"
    primary_effect_interpretation = "Mean within-subject change in the selected edge delta between the two selected timepoints."

    if between_groups:
        selected_group_b = str(analysis_result.get("selected_group_b"))
        selected_group_a = str(analysis_result.get("selected_group_a"))
        frame["group_indicator"] = (frame["group_label"].astype(str) == selected_group_b).astype(float)
        frame["interaction"] = frame["group_indicator"] * frame["timepoint_indicator"]
        predictor_columns = ["group_indicator", "timepoint_indicator", "interaction"]
        coefficient_labels.update(
            {
                "group_indicator": f"Group main effect ({selected_group_b} vs {selected_group_a})",
                "interaction": (
                    f"Group x timepoint interaction ({selected_group_b} vs {selected_group_a}) x "
                    f"({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})"
                ),
            }
        )
        primary_effect_name = "interaction"
        primary_effect_interpretation = "Difference between groups in the longitudinal change of the selected edge delta."

    usable_covariates = [column for column in covariates if frame[column].nunique(dropna=True) > 1]
    required_columns = ["trial_delta", "idx"] + predictor_columns + usable_covariates
    frame = frame.dropna(subset=required_columns).copy()
    if len(frame) < max(6, len(predictor_columns) + len(usable_covariates) + 2):
        return {
            "message": "Not enough complete repeated observations for a mixed-effects model.",
            "n": int(len(frame)),
        }

    exog_columns = predictor_columns + usable_covariates
    exog = sm.add_constant(frame[exog_columns], has_constant="add")
    endog = frame["trial_delta"].astype(float)
    groups = frame["idx"].astype(str)

    ordered_timepoints = frame["timepoint_order"].dropna().nunique()
    random_slope_used = bool(random_slope_time and ordered_timepoints >= 3 and frame.groupby("idx")["timepoint_label"].nunique().max() >= 3)
    model_note = None
    try:
        if random_slope_used:
            exog_re = sm.add_constant(frame[["timepoint_order"]].astype(float), has_constant="add")
            model = sm.MixedLM(endog, exog, groups=groups, exog_re=exog_re).fit(reml=False, method="lbfgs", disp=False)
        else:
            if random_slope_time:
                model_note = "Random slope requested, but the selected comparison does not contain enough repeated timepoints. Falling back to a random-intercept model."
            model = sm.MixedLM(endog, exog, groups=groups).fit(reml=False, method="lbfgs", disp=False)
    except Exception as exc:
        return {
            "message": f"Mixed-effects estimation failed for the selected edge: {exc}",
            "n": int(len(frame)),
        }

    confidence = model.conf_int()
    coefficient_rows = []
    for name in model.fe_params.index:
        coefficient_rows.append(
            {
                "name": name,
                "label": coefficient_labels.get(name, name),
                "estimate": float(model.fe_params[name]),
                "std_error": float(model.bse_fe[name]),
                "t_value": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
                "ci_low": float(confidence.loc[name, 0]),
                "ci_high": float(confidence.loc[name, 1]),
            }
        )

    point_labels = _build_longitudinal_point_labels(frame)
    observed = endog.to_numpy(dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = observed - fitted
    diagnostics = {
        "observed": observed.tolist(),
        "fitted": fitted.tolist(),
        "residuals": residuals.tolist(),
        "point_labels": point_labels,
        "message": model_note,
    }

    primary_effect = None
    if primary_effect_name in model.fe_params.index:
        primary_effect = {
            "name": primary_effect_name,
            "label": coefficient_labels.get(primary_effect_name, primary_effect_name),
            "estimate": float(model.fe_params[primary_effect_name]),
            "std_error": float(model.bse_fe[primary_effect_name]),
            "t_value": float(model.tvalues[primary_effect_name]),
            "p_value": float(model.pvalues[primary_effect_name]),
            "ci_low": float(confidence.loc[primary_effect_name, 0]),
            "ci_high": float(confidence.loc[primary_effect_name, 1]),
            "interpretation": primary_effect_interpretation,
        }

    variance_components = {
        "random_intercept_variance": float(model.cov_re.iloc[0, 0]) if getattr(model, "cov_re", None) is not None and model.cov_re.shape[0] >= 1 else np.nan,
        "residual_variance": float(model.scale) if model.scale is not None else np.nan,
    }
    if getattr(model, "cov_re", None) is not None and model.cov_re.shape[0] >= 2:
        variance_components["random_slope_variance"] = float(model.cov_re.iloc[1, 1])

    formula = "trial_delta ~ " + " + ".join(exog_columns) if exog_columns else "trial_delta ~ 1"
    return {
        "message": None,
        "model_label": "Mixed-effects longitudinal model",
        "formula": formula,
        "response_definition": "Timepoint-level selected-edge trial delta",
        "observation_unit": "One row per subject-timepoint measurement for the selected edge",
        "methods_description": (
            "This model estimates repeated-measure longitudinal effects on the selected edge using a mixed-effects model "
            "with a subject-level random intercept and fixed effects for the selected timepoints, group setting, and optional covariates."
        ),
        "n": int(len(frame)),
        "variables": usable_covariates,
        "coefficients": coefficient_rows,
        "primary_effect": primary_effect,
        "variance_components": variance_components,
        "random_slope_used": bool(random_slope_used),
        "diagnostics": diagnostics,
    }


def run_change_score_analysis(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    regression_covariates: list[str] | None = None,
    baseline_value: Any = None,
    followup_value: Any = None,
) -> dict[str, Any]:
    """Fit a change-score model for the selected edge."""
    paired = _build_paired_longitudinal_frame(
        edge_frame,
        analysis_result,
        baseline_value=baseline_value,
        followup_value=followup_value,
        regression_covariates=regression_covariates,
    )
    if paired.get("message"):
        return paired
    frame = paired["frame"].copy()
    response_column = "change_score"
    frame[response_column] = frame["followup_delta"] - frame["baseline_delta"]

    predictors: list[str] = []
    coefficient_labels = {"const": "Intercept"}
    primary_effect_name = "const"
    primary_effect_interpretation = "Mean longitudinal change score in the selected edge."
    if paired["between_groups"]:
        predictors.append("group_indicator")
        coefficient_labels["group_indicator"] = f"Group effect ({paired['group_b']} vs {paired['group_a']})"
        primary_effect_name = "group_indicator"
        primary_effect_interpretation = "Adjusted between-group difference in selected-edge change scores."
    for covariate in paired["covariates"]:
        if frame[covariate].nunique(dropna=True) > 1:
            predictors.append(covariate)
            coefficient_labels[covariate] = covariate

    return _fit_longitudinal_ols_model(
        frame=frame,
        response_column=response_column,
        predictors=predictors,
        coefficient_labels=coefficient_labels,
        model_label="Change-score longitudinal model",
        response_definition="Selected-edge follow-up delta minus baseline delta",
        observation_unit="One row per paired subject for the selected edge",
        methods_description=(
            "This robustness analysis models the subject-level change score on the selected edge, defined as "
            "follow-up trial delta minus baseline trial delta."
        ),
        primary_effect_name=primary_effect_name,
        primary_effect_interpretation=primary_effect_interpretation,
    )


def run_ancova_analysis(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    regression_covariates: list[str] | None = None,
    baseline_value: Any = None,
    followup_value: Any = None,
) -> dict[str, Any]:
    """Fit a baseline-adjusted ANCOVA model for the selected edge."""
    paired = _build_paired_longitudinal_frame(
        edge_frame,
        analysis_result,
        baseline_value=baseline_value,
        followup_value=followup_value,
        regression_covariates=regression_covariates,
    )
    if paired.get("message"):
        return paired
    frame = paired["frame"].copy()
    predictors = ["baseline_delta"]
    coefficient_labels = {"const": "Intercept", "baseline_delta": "Baseline selected-edge delta"}
    primary_effect_name = None
    primary_effect_interpretation = "Adjusted follow-up estimate for the selected edge after controlling for baseline."
    if paired["between_groups"]:
        predictors.insert(0, "group_indicator")
        coefficient_labels["group_indicator"] = f"Group effect ({paired['group_b']} vs {paired['group_a']})"
        primary_effect_name = "group_indicator"
        primary_effect_interpretation = "Adjusted between-group difference in follow-up selected-edge delta after controlling for baseline."
    for covariate in paired["covariates"]:
        if frame[covariate].nunique(dropna=True) > 1:
            predictors.append(covariate)
            coefficient_labels[covariate] = covariate

    return _fit_longitudinal_ols_model(
        frame=frame,
        response_column="followup_delta",
        predictors=predictors,
        coefficient_labels=coefficient_labels,
        model_label="ANCOVA longitudinal model",
        response_definition="Selected-edge follow-up trial delta",
        observation_unit="One row per paired subject for the selected edge",
        methods_description=(
            "This robustness analysis models the follow-up selected-edge delta while adjusting for the baseline selected-edge delta "
            "and optional covariates."
        ),
        primary_effect_name=primary_effect_name,
        primary_effect_interpretation=primary_effect_interpretation,
    )


def run_edge_trajectory_analysis(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Prepare selected-edge subject trajectories and group trends."""
    if edge_frame.empty:
        return {"message": "No trajectory data are available for the selected edge."}

    frame = edge_frame.copy().sort_values(["timepoint_order", "idx", "subject_id"])
    timepoint_order = frame[["timepoint_label", "timepoint_order"]].drop_duplicates().sort_values("timepoint_order")
    ordered_labels = timepoint_order["timepoint_label"].astype(str).tolist()

    traces = []
    for idx, group in frame.groupby("idx", sort=False):
        ordered = group.sort_values("timepoint_order")
        traces.append(
            {
                "idx": str(idx),
                "group_label": str(ordered["group_label"].iloc[0]),
                "x": ordered["timepoint_label"].astype(str).tolist(),
                "y": ordered["trial_delta"].astype(float).tolist(),
            }
        )

    group_trends = []
    alpha = float(np.clip(ci_level, 0.5, 0.999))
    for group_name, group in frame.groupby("group_label", sort=False):
        summary_rows = []
        for label in ordered_labels:
            values = group[group["timepoint_label"].astype(str) == label]["trial_delta"].to_numpy(dtype=float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            if n >= 2:
                sem = float(stats.sem(values, nan_policy="omit"))
                critical = float(stats.t.ppf((1 + alpha) / 2, n - 1))
                lower = mean - critical * sem
                upper = mean + critical * sem
            else:
                lower = mean
                upper = mean
            summary_rows.append({"timepoint_label": label, "mean": mean, "lower": lower, "upper": upper, "n": int(n)})
        group_trends.append({"group_label": str(group_name), "rows": summary_rows})

    return {
        "message": None,
        "ordered_timepoints": ordered_labels,
        "subject_traces": traces,
        "group_trends": group_trends,
        "selected_pair": [str(analysis_result["longitudinal_value_a"]), str(analysis_result["longitudinal_value_b"])],
        "uses_all_available_timepoints": True,
    }


def run_reliable_change_analysis(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    baseline_value: Any = None,
    followup_value: Any = None,
    method: str = "jt_baseline_sd",
    reliability_coefficient: float = 0.8,
) -> dict[str, Any]:
    """Estimate reliable change for the selected edge using paired observations."""
    paired = _build_paired_longitudinal_frame(
        edge_frame,
        analysis_result,
        baseline_value=baseline_value,
        followup_value=followup_value,
        regression_covariates=None,
    )
    if paired.get("message"):
        return paired
    frame = paired["frame"].copy()
    if len(frame) < 2:
        return {"message": "Reliable change requires at least two paired subjects."}

    baseline_sd = float(frame["baseline_delta"].std(ddof=1))
    reliability = float(np.clip(reliability_coefficient, 0.0, 0.99))
    sdiff = baseline_sd * float(np.sqrt(max(2 * (1 - reliability), 0.0)))
    if not pd.notna(sdiff) or sdiff <= 0:
        return {"message": "Reliable change is not estimable because baseline variability is too small."}

    frame["change_score"] = frame["followup_delta"] - frame["baseline_delta"]
    frame["rci"] = frame["change_score"] / sdiff
    frame["category"] = np.where(
        frame["rci"] >= 1.96,
        "reliable increase",
        np.where(frame["rci"] <= -1.96, "reliable decrease", "no reliable change"),
    )

    category_counts = frame["category"].value_counts().reindex(
        ["reliable increase", "no reliable change", "reliable decrease"],
        fill_value=0,
    )
    group_summaries = []
    for group_name, group in frame.groupby("group_label", sort=False):
        counts = group["category"].value_counts().reindex(
            ["reliable increase", "no reliable change", "reliable decrease"],
            fill_value=0,
        )
        group_summaries.append(
            {
                "group_label": str(group_name),
                "n": int(len(group)),
                "reliable_increase": int(counts["reliable increase"]),
                "no_reliable_change": int(counts["no reliable change"]),
                "reliable_decrease": int(counts["reliable decrease"]),
            }
        )

    records = []
    for _, row in frame.sort_values(["group_label", "idx"]).iterrows():
        records.append(
            {
                "idx": str(row["idx"]),
                "group_label": str(row["group_label"]),
                "baseline_delta": float(row["baseline_delta"]),
                "followup_delta": float(row["followup_delta"]),
                "change_score": float(row["change_score"]),
                "rci": float(row["rci"]),
                "category": str(row["category"]),
            }
        )

    return {
        "message": None,
        "method": method,
        "method_label": "Jacobson-Truax style approximation using baseline SD and assumed reliability",
        "reliability_coefficient": reliability,
        "baseline_sd": baseline_sd,
        "sdiff": sdiff,
        "records": records,
        "category_counts": {key: int(value) for key, value in category_counts.to_dict().items()},
        "group_summaries": group_summaries,
    }


def run_leave_one_out_analysis(
    bundle: DatasetBundle,
    config: AnalysisConfig,
    pair_key: str,
    *,
    significance_threshold: float,
    outcome_variable: str | None = None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Run leave-one-out influence analysis for the current configuration."""
    baseline_analysis = run_analysis(bundle, config)
    baseline_pair = summarize_pair_result(baseline_analysis, pair_key)
    baseline_regression = run_multivariate_regression_analysis(
        bundle,
        baseline_analysis,
        pair_key,
        outcome_variable=outcome_variable,
        regression_covariates=regression_covariates,
    )
    excluded_idx = set(_normalize_excluded_idx(config.excluded_idx))
    candidate_idx = sorted(
        str(idx)
        for idx in bundle.subjects["idx"].dropna().astype(str).unique().tolist()
        if str(idx) not in excluded_idx
    )

    global_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    regression_records: list[dict[str, Any]] = []
    for idx in candidate_idx:
        loo_config = replace(config, excluded_idx=tuple(sorted(excluded_idx | {idx})))
        loo_analysis = run_analysis(bundle, loo_config)
        loo_pair = summarize_pair_result(loo_analysis, pair_key)
        loo_regression = run_multivariate_regression_analysis(
            bundle,
            loo_analysis,
            pair_key,
            outcome_variable=outcome_variable,
            regression_covariates=regression_covariates,
        )
        global_records.append(
            _build_global_influence_record(
                baseline_analysis,
                loo_analysis,
                excluded_idx=idx,
                significance_threshold=significance_threshold,
            )
        )
        pair_records.append(
            _build_pair_influence_record(
                baseline_pair,
                loo_pair,
                excluded_idx=idx,
                significance_threshold=significance_threshold,
            )
        )
        regression_records.append(
            _build_regression_change_record(
                baseline_regression,
                loo_regression,
                excluded_idx=idx,
            )
        )

    global_records = sorted(
        global_records,
        key=lambda row: (
            -(row["significance_loss_count"] + row["significance_gain_count"]),
            -row["mean_abs_effect_size_change"],
            -row["mean_abs_mean_delta_change"],
            str(row["excluded_idx"]),
        ),
    )
    pair_records = sorted(
        pair_records,
        key=lambda row: (
            -int(row["significance_switched"]),
            -row["abs_effect_size_change"],
            -row["abs_mean_delta_change"],
            -row["abs_q_value_change"],
            str(row["excluded_idx"]),
        ),
    )
    regression_records = sorted(
        regression_records,
        key=lambda row: (
            -int(row["model_valid"]),
            -row["abs_primary_effect_change"],
            str(row["excluded_idx"]),
        ),
    )

    return {
        "baseline_analysis": baseline_analysis,
        "baseline_pair": baseline_pair,
        "baseline_regression": baseline_regression,
        "global_records": global_records,
        "pair_records": pair_records,
        "regression_records": regression_records,
        "top3_idx": [record["excluded_idx"] for record in global_records[:3]],
    }


def _build_edge_longitudinal_frame(bundle: DatasetBundle, analysis_result: dict[str, Any], pair_key: str) -> pd.DataFrame:
    if "|" not in pair_key:
        return pd.DataFrame()
    roi_from, roi_to = pair_key.split("|", 1)
    dataset_id = str(analysis_result.get("dataset_id", bundle.dataset_id))
    metric = str(analysis_result.get("metric", bundle.metric))
    trial_a = analysis_result.get("trial_a")
    trial_b = analysis_result.get("trial_b")
    if trial_a is None or trial_b is None:
        return pd.DataFrame()

    frame = bundle.connectivity[
        (bundle.connectivity["dataset_id"] == dataset_id)
        & (bundle.connectivity["metric"] == metric)
        & (bundle.connectivity["freq"] >= float(analysis_result.get("freq_min", np.nan)))
        & (bundle.connectivity["freq"] <= float(analysis_result.get("freq_max", np.nan)))
        & (bundle.connectivity["roi_from"] == roi_from)
        & (bundle.connectivity["roi_to"] == roi_to)
        & (bundle.connectivity["trial_id"].isin([trial_a, trial_b]))
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    excluded_idx = set(_normalize_excluded_idx(analysis_result.get("excluded_idx", ())))
    if excluded_idx:
        frame = frame[~frame["idx"].astype(str).isin(excluded_idx)].copy()

    selected_group_a = str(analysis_result.get("selected_group_a"))
    selected_group_b = str(analysis_result.get("selected_group_b"))
    allowed_groups = None
    if selected_group_a != ALL_GROUPS_LABEL and selected_group_b != ALL_GROUPS_LABEL:
        allowed_groups = {selected_group_a}
        if analysis_result.get("between_groups"):
            allowed_groups.add(selected_group_b)
    if allowed_groups:
        frame = frame[frame["group_label"].astype(str).isin(allowed_groups)].copy()

    frame = (
        frame.groupby(
            ["dataset_id", "metric", "subject_id", "idx", "base_subject_id", "group_label", "mtime", "trial_id", "roi_from", "roi_to"],
            as_index=False,
        )["value"]
        .mean()
    )
    frame["timepoint_label"] = frame["mtime"].astype(str)
    order_lookup = _timepoint_order_lookup(frame["timepoint_label"].dropna().astype(str).unique().tolist())
    frame["timepoint_order"] = frame["timepoint_label"].map(order_lookup).astype(float)

    pivoted = (
        frame.pivot_table(
            index=["subject_id", "idx", "base_subject_id", "group_label", "mtime", "timepoint_label", "timepoint_order", "roi_from", "roi_to"],
            columns="trial_id",
            values="value",
        )
        .dropna(subset=[trial_a, trial_b])
        .reset_index()
        .rename(columns={trial_a: "trial_a_value", trial_b: "trial_b_value"})
    )
    if pivoted.empty:
        return pd.DataFrame()
    pivoted["trial_delta"] = pivoted["trial_b_value"] - pivoted["trial_a_value"]

    subject_columns = [column for column in bundle.subjects.columns if column not in pivoted.columns or column == "subject_id"]
    merged = pivoted.merge(bundle.subjects[subject_columns], on="subject_id", how="left")
    return merged.sort_values(["timepoint_order", "group_label", "idx", "subject_id"]).reset_index(drop=True)


def _selected_timepoint_frame(edge_frame: pd.DataFrame, analysis_result: dict[str, Any]) -> pd.DataFrame:
    if edge_frame.empty:
        return edge_frame
    time_a = str(analysis_result.get("longitudinal_value_a"))
    time_b = str(analysis_result.get("longitudinal_value_b"))
    return edge_frame[edge_frame["timepoint_label"].astype(str).isin([time_a, time_b])].copy()


def _build_paired_longitudinal_frame(
    edge_frame: pd.DataFrame,
    analysis_result: dict[str, Any],
    *,
    baseline_value: Any = None,
    followup_value: Any = None,
    regression_covariates: list[str] | None = None,
) -> dict[str, Any]:
    if edge_frame.empty:
        return {"message": "No longitudinal observations are available for the selected edge."}
    time_a = str(baseline_value if baseline_value is not None else analysis_result.get("longitudinal_value_a"))
    time_b = str(followup_value if followup_value is not None else analysis_result.get("longitudinal_value_b"))
    selected = edge_frame[edge_frame["timepoint_label"].astype(str).isin([time_a, time_b])].copy()
    if selected.empty:
        return {"message": "No paired longitudinal observations are available for the selected edge."}

    covariates = [column for column in (regression_covariates or []) if column in selected.columns]
    value_columns = ["subject_id", "trial_delta"] + covariates
    pivoted = selected.pivot_table(
        index=["idx", "base_subject_id", "group_label"],
        columns="timepoint_label",
        values=value_columns,
        aggfunc="first",
    )
    required = [("trial_delta", time_a), ("trial_delta", time_b)]
    if not all(column in pivoted.columns for column in required):
        return {"message": "Change-score and reliable-change analyses require paired subjects at both selected timepoints."}
    pivoted = pivoted.dropna(subset=required).reset_index()
    if pivoted.empty:
        return {"message": "No paired subjects are available at both selected timepoints for the selected edge."}

    paired = pd.DataFrame(
        {
            "idx": pivoted["idx"].astype(str),
            "base_subject_id": pivoted["base_subject_id"].astype(str),
            "group_label": pivoted["group_label"].astype(str),
            "baseline_delta": pivoted[("trial_delta", time_a)].astype(float),
            "followup_delta": pivoted[("trial_delta", time_b)].astype(float),
            "subject_id_a": pivoted[("subject_id", time_a)].astype(str) if ("subject_id", time_a) in pivoted.columns else pivoted["idx"].astype(str),
            "subject_id_b": pivoted[("subject_id", time_b)].astype(str) if ("subject_id", time_b) in pivoted.columns else pivoted["idx"].astype(str),
        }
    )
    for covariate in covariates:
        if (covariate, time_a) in pivoted.columns:
            paired[covariate] = pivoted[(covariate, time_a)]

    between_groups = bool(analysis_result.get("between_groups"))
    selected_group_a = str(analysis_result.get("selected_group_a"))
    selected_group_b = str(analysis_result.get("selected_group_b"))
    if between_groups:
        paired["group_indicator"] = (paired["group_label"].astype(str) == selected_group_b).astype(float)

    return {
        "message": None,
        "frame": paired,
        "baseline_label": time_a,
        "followup_label": time_b,
        "between_groups": between_groups,
        "group_a": selected_group_a,
        "group_b": selected_group_b,
        "covariates": [column for column in covariates if column in paired.columns],
    }


def _fit_longitudinal_ols_model(
    *,
    frame: pd.DataFrame,
    response_column: str,
    predictors: list[str],
    coefficient_labels: dict[str, str],
    model_label: str,
    response_definition: str,
    observation_unit: str,
    methods_description: str,
    primary_effect_name: str | None,
    primary_effect_interpretation: str | None,
) -> dict[str, Any]:
    required = [response_column] + predictors
    regression_frame = frame.dropna(subset=required).copy()
    min_required = max(3, len(predictors) + 2)
    if len(regression_frame) < min_required:
        return {
            "message": "Not enough complete paired observations for this longitudinal model.",
            "n": int(len(regression_frame)),
        }

    exog = sm.add_constant(regression_frame[predictors], has_constant="add") if predictors else pd.DataFrame({"const": np.ones(len(regression_frame))}, index=regression_frame.index)
    model = sm.OLS(regression_frame[response_column].astype(float), exog).fit()
    confidence = model.conf_int()
    coefficient_rows = []
    for name in model.params.index:
        coefficient_rows.append(
            {
                "name": name,
                "label": coefficient_labels.get(name, name),
                "estimate": float(model.params[name]),
                "std_error": float(model.bse[name]),
                "t_value": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
                "ci_low": float(confidence.loc[name, 0]),
                "ci_high": float(confidence.loc[name, 1]),
            }
        )

    point_labels = [
        f"{str(row['idx'])} | {str(row.get('group_label', 'n/a'))}"
        for _, row in regression_frame.iterrows()
    ]
    diagnostics = {
        "observed": regression_frame[response_column].astype(float).tolist(),
        "fitted": model.fittedvalues.astype(float).tolist(),
        "residuals": model.resid.astype(float).tolist(),
        "point_labels": point_labels,
        "message": "Diagnostics are exploratory because this robustness model is fit on paired summaries." if len(regression_frame) < max(6, len(predictors) + 3) else None,
    }

    primary_effect = None
    if primary_effect_name and primary_effect_name in model.params.index:
        primary_effect = {
            "name": primary_effect_name,
            "label": coefficient_labels.get(primary_effect_name, primary_effect_name),
            "estimate": float(model.params[primary_effect_name]),
            "std_error": float(model.bse[primary_effect_name]),
            "t_value": float(model.tvalues[primary_effect_name]),
            "p_value": float(model.pvalues[primary_effect_name]),
            "ci_low": float(confidence.loc[primary_effect_name, 0]),
            "ci_high": float(confidence.loc[primary_effect_name, 1]),
            "interpretation": primary_effect_interpretation,
        }

    return {
        "message": None,
        "model_label": model_label,
        "formula": _build_formula(response_definition, [], predictors),
        "response_definition": response_definition,
        "observation_unit": observation_unit,
        "methods_description": methods_description,
        "n": int(len(regression_frame)),
        "variables": predictors,
        "coefficients": coefficient_rows,
        "primary_effect": primary_effect,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "diagnostics": diagnostics,
    }


def _timepoint_order_lookup(labels: list[str]) -> dict[str, int]:
    def sort_key(label: str) -> tuple[int, str]:
        digits = "".join(character for character in str(label) if character.isdigit())
        if digits:
            return (int(digits), str(label))
        return (10_000, str(label))

    return {label: index for index, label in enumerate(sorted(labels, key=sort_key))}


def _prepare_regression_frame(bundle: DatasetBundle, analysis_result: dict[str, Any], detail_frame: pd.DataFrame) -> dict[str, Any]:
    selected_group_a = analysis_result["selected_group_a"]
    selected_group_b = analysis_result["selected_group_b"]
    between_groups = analysis_result["between_groups"]

    if analysis_result["longitudinal_enabled"]:
        if {"measure_a_value", "measure_b_value", "delta_a", "delta_b"}.issubset(detail_frame.columns):
            baseline = (
                bundle.subjects.assign(mtime_order=bundle.subjects["mtime"].str.extract(r"(\d+)").fillna("999").astype(int))
                .sort_values(["base_subject_id", "mtime_order"])
                .drop_duplicates("base_subject_id", keep="first")
                .drop(columns="mtime_order")
            )
            baseline_columns = [column for column in baseline.columns if column not in detail_frame.columns or column == "base_subject_id"]
            merged = detail_frame.merge(baseline[baseline_columns], on="base_subject_id", how="left")
            merged["selected_edge_delta"] = merged["delta"].astype(float)
            if between_groups:
                merged["group_indicator"] = (merged["group_label"].astype(str) == str(selected_group_b)).astype(int)
                design_columns = ["group_indicator"]
                coefficient_labels = {
                    "const": "Intercept",
                    "group_indicator": f"Group effect ({selected_group_b} vs {selected_group_a})",
                }
                primary_effect_name = "group_indicator"
                primary_effect_label = coefficient_labels["group_indicator"]
                primary_effect_interpretation = "Adjusted difference between groups in subject-level paired longitudinal delta differences."
                model_label = "Paired longitudinal regression with group adjustment"
            else:
                design_columns = []
                coefficient_labels = {"const": "Intercept"}
                primary_effect_name = None
                primary_effect_label = None
                primary_effect_interpretation = "No explicit design term is included in this one-group paired longitudinal model."
                model_label = "Paired longitudinal regression"

            return {
                "frame": merged,
                "design_columns": design_columns,
                "coefficient_labels": coefficient_labels,
                "primary_effect_name": primary_effect_name,
                "primary_effect_label": primary_effect_label,
                "primary_effect_interpretation": primary_effect_interpretation,
                "model_label": model_label,
                "default_response_definition": "Subject-level paired longitudinal delta difference",
                "observation_unit": "One row per subject with measurements at both selected timepoints",
                "methods_description": (
                    "The response variable is the subject-level paired longitudinal delta difference: "
                    "for each subject and ROI pair, the trial delta at the second selected timepoint minus the trial delta at the first selected timepoint. "
                    + (
                        f"The model includes an explicit group term ({selected_group_b} vs {selected_group_a}) to estimate the adjusted between-group difference in this paired longitudinal contrast."
                        if between_groups
                        else "The model regresses this paired longitudinal contrast on the selected covariates within the selected analysis set."
                    )
                ),
            }

        subject_columns = [column for column in bundle.subjects.columns if column not in detail_frame.columns or column == "subject_id"]
        merged = detail_frame.merge(bundle.subjects[subject_columns], on="subject_id", how="left")
        merged["selected_edge_delta"] = merged["delta"].astype(float)
        merged["timepoint_indicator"] = (
            merged["timepoint_label"].astype(str) == str(analysis_result["longitudinal_value_b"])
        ).astype(int)

        if between_groups:
            merged["group_indicator"] = (merged["group_label"].astype(str) == str(selected_group_b)).astype(int)
            merged["interaction"] = merged["group_indicator"] * merged["timepoint_indicator"]
            design_columns = ["group_indicator", "timepoint_indicator", "interaction"]
            coefficient_labels = {
                "const": "Intercept",
                "group_indicator": f"Group main effect ({selected_group_b} vs {selected_group_a})",
                "timepoint_indicator": (
                    f"Timepoint main effect ({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})"
                ),
                "interaction": (
                    f"Group x timepoint interaction ({selected_group_b} vs {selected_group_a}) x "
                    f"({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})"
                ),
            }
            primary_effect_name = "interaction"
            primary_effect_label = coefficient_labels["interaction"]
            primary_effect_interpretation = "Adjusted group-by-timepoint interaction on timepoint-level trial deltas."
            model_label = "Unpaired longitudinal interaction regression"
        else:
            design_columns = ["timepoint_indicator"]
            coefficient_labels = {
                "const": "Intercept",
                "timepoint_indicator": (
                    f"Timepoint effect ({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})"
                ),
            }
            primary_effect_name = "timepoint_indicator"
            primary_effect_label = coefficient_labels["timepoint_indicator"]
            primary_effect_interpretation = "Adjusted difference between the selected timepoints in timepoint-level trial deltas."
            model_label = "Unpaired longitudinal regression"

        return {
            "frame": merged,
            "design_columns": design_columns,
            "coefficient_labels": coefficient_labels,
            "primary_effect_name": primary_effect_name,
            "primary_effect_label": primary_effect_label,
            "primary_effect_interpretation": primary_effect_interpretation,
            "model_label": model_label,
            "default_response_definition": "Timepoint-level trial delta",
            "observation_unit": "One row per subject-timepoint measurement",
            "methods_description": (
                "The response variable is the timepoint-level trial delta computed for each available subject-timepoint measurement and ROI pair. "
                + (
                    f"The model includes explicit group, timepoint, and group-by-timepoint terms to estimate the adjusted interaction between group ({selected_group_b} vs {selected_group_a}) "
                    f"and timepoint ({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']})."
                    if between_groups
                    else f"The model includes an explicit timepoint term ({analysis_result['longitudinal_value_b']} vs {analysis_result['longitudinal_value_a']}) to estimate the adjusted longitudinal effect."
                )
            ),
        }

    subject_columns = [column for column in bundle.subjects.columns if column not in detail_frame.columns or column == "subject_id"]
    merged = detail_frame.merge(bundle.subjects[subject_columns], on="subject_id", how="left")
    merged["selected_edge_delta"] = merged["delta"].astype(float)

    if between_groups:
        merged["group_indicator"] = (merged["group_label"].astype(str) == str(selected_group_b)).astype(int)
        design_columns = ["group_indicator"]
        coefficient_labels = {
            "const": "Intercept",
            "group_indicator": f"Group effect ({selected_group_b} vs {selected_group_a})",
        }
        primary_effect_name = "group_indicator"
        primary_effect_label = coefficient_labels["group_indicator"]
        primary_effect_interpretation = "Adjusted difference between groups in subject-level trial deltas."
        model_label = "Trial-delta regression with group adjustment"
    else:
        design_columns = []
        coefficient_labels = {"const": "Intercept"}
        primary_effect_name = None
        primary_effect_label = None
        primary_effect_interpretation = "No explicit design term is included in this one-group trial-delta model."
        model_label = "Trial-delta regression"

    return {
        "frame": merged,
        "design_columns": design_columns,
        "coefficient_labels": coefficient_labels,
        "primary_effect_name": primary_effect_name,
        "primary_effect_label": primary_effect_label,
        "primary_effect_interpretation": primary_effect_interpretation,
        "model_label": model_label,
        "default_response_definition": "Subject-level trial delta (Trial B - Trial A)",
        "observation_unit": "One row per subject",
        "methods_description": (
            "The response variable is the subject-level trial delta computed for each subject and ROI pair as Trial B minus Trial A. "
            + (
                f"The model includes an explicit group term ({selected_group_b} vs {selected_group_a}) to estimate the adjusted between-group difference in this trial delta."
                if between_groups
                else "The model regresses the subject-level trial delta on the selected covariates within the selected analysis set."
            )
        ),
    }


def _build_formula(response_name: str, design_columns: list[str], covariates: list[str]) -> str:
    predictors = design_columns + list(covariates)
    if not predictors:
        return f"{response_name} ~ 1"
    return f"{response_name} ~ " + " + ".join(predictors)


def _build_regression_point_labels(frame: pd.DataFrame) -> list[str]:
    if {"idx", "mtime"}.issubset(frame.columns):
        return (frame["idx"].astype(str) + "-" + frame["mtime"].astype(str)).tolist()
    if "idx" in frame.columns:
        return frame["idx"].astype(str).tolist()
    if "subject_id" in frame.columns:
        return frame["subject_id"].astype(str).tolist()
    if "base_subject_id" in frame.columns:
        return frame["base_subject_id"].astype(str).tolist()
    return [str(index) for index in range(len(frame))]


def _build_longitudinal_point_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in frame.iterrows():
        idx = str(row["idx"]) if "idx" in frame.columns and pd.notna(row["idx"]) else "n/a"
        group = str(row["group_label"]) if "group_label" in frame.columns and pd.notna(row["group_label"]) else "n/a"
        timepoint = str(row["timepoint_label"]) if "timepoint_label" in frame.columns and pd.notna(row["timepoint_label"]) else "n/a"
        labels.append(f"{idx} | {group} | {timepoint}")
    return labels


def _build_regression_observation_rows(
    *,
    regression_frame: pd.DataFrame,
    point_labels: list[str],
    observed: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    leverage: np.ndarray,
    cooks_distance: np.ndarray,
    studentized_residuals: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(point_labels):
        row: dict[str, Any] = {
            "label": label,
            "observed": float(observed[index]),
            "fitted": float(fitted[index]),
            "residual": float(residuals[index]),
            "leverage": float(leverage[index]),
            "cooks_distance": float(cooks_distance[index]),
            "studentized_residual": float(studentized_residuals[index]),
        }
        for column in ("idx", "subject_id", "mtime", "base_subject_id", "subject_id_a", "subject_id_b", "group_label", "timepoint_label"):
            if column in regression_frame.columns and pd.notna(regression_frame.iloc[index][column]):
                row[column] = str(regression_frame.iloc[index][column])
        rows.append(row)
    return rows


def _normalize_excluded_idx(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(sorted({str(value) for value in values if value is not None and str(value) != ""}))


def _record_lookup(analysis_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["pair_key"]: record for record in analysis_result.get("pair_results", [])}


def _significance_value(record: dict[str, Any], correction_method: str) -> float:
    if correction_method == "none":
        return float(record.get("p_value", np.nan))
    return float(record.get("q_value", np.nan))


def _is_significant(record: dict[str, Any] | None, correction_method: str, significance_threshold: float) -> bool:
    if not record:
        return False
    value = _significance_value(record, correction_method)
    return pd.notna(value) and float(value) < float(significance_threshold)


def _float_or_nan(value: Any) -> float:
    return float(value) if value is not None and pd.notna(value) else np.nan


def _coefficient_lookup(regression_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in regression_result.get("coefficients", [])}


def _build_global_influence_record(
    baseline_analysis: dict[str, Any],
    loo_analysis: dict[str, Any],
    *,
    excluded_idx: str,
    significance_threshold: float,
) -> dict[str, Any]:
    baseline_lookup = _record_lookup(baseline_analysis)
    loo_lookup = _record_lookup(loo_analysis)
    mean_delta_changes: list[float] = []
    effect_size_changes: list[float] = []
    significance_loss_count = 0
    significance_gain_count = 0
    significance_switch_count = 0
    correction_method = str(baseline_analysis["correction_method"])

    for pair_key, baseline_record in baseline_lookup.items():
        loo_record = loo_lookup.get(pair_key)
        mean_delta_change = _float_or_nan(loo_record.get("mean_delta")) - _float_or_nan(baseline_record.get("mean_delta")) if loo_record else np.nan
        effect_size_change = _float_or_nan(loo_record.get("effect_size")) - _float_or_nan(baseline_record.get("effect_size")) if loo_record else np.nan
        if pd.notna(mean_delta_change):
            mean_delta_changes.append(abs(float(mean_delta_change)))
        if pd.notna(effect_size_change):
            effect_size_changes.append(abs(float(effect_size_change)))

        baseline_sig = _is_significant(baseline_record, correction_method, significance_threshold)
        loo_sig = _is_significant(loo_record, correction_method, significance_threshold)
        if baseline_sig and not loo_sig:
            significance_loss_count += 1
        elif not baseline_sig and loo_sig:
            significance_gain_count += 1
        if baseline_sig != loo_sig:
            significance_switch_count += 1

    return {
        "excluded_idx": str(excluded_idx),
        "mean_abs_mean_delta_change": float(np.mean(mean_delta_changes)) if mean_delta_changes else 0.0,
        "mean_abs_effect_size_change": float(np.mean(effect_size_changes)) if effect_size_changes else 0.0,
        "significance_loss_count": int(significance_loss_count),
        "significance_gain_count": int(significance_gain_count),
        "significance_switch_count": int(significance_switch_count),
    }


def _build_pair_influence_record(
    baseline_pair: dict[str, Any] | None,
    loo_pair: dict[str, Any] | None,
    *,
    excluded_idx: str,
    significance_threshold: float,
) -> dict[str, Any]:
    baseline_pair = baseline_pair or {}
    loo_pair = loo_pair or {}
    correction_method = "none" if baseline_pair.get("q_value") is None and baseline_pair.get("q_value_none") is None else "q"
    baseline_sig = False
    loo_sig = False
    if baseline_pair:
        baseline_value = _float_or_nan(baseline_pair.get("q_value", baseline_pair.get("p_value")))
        baseline_sig = pd.notna(baseline_value) and baseline_value < significance_threshold
    if loo_pair:
        loo_value = _float_or_nan(loo_pair.get("q_value", loo_pair.get("p_value")))
        loo_sig = pd.notna(loo_value) and loo_value < significance_threshold
    mean_delta_change = _float_or_nan(loo_pair.get("mean_delta")) - _float_or_nan(baseline_pair.get("mean_delta"))
    effect_size_change = _float_or_nan(loo_pair.get("effect_size")) - _float_or_nan(baseline_pair.get("effect_size"))
    p_value_change = _float_or_nan(loo_pair.get("p_value")) - _float_or_nan(baseline_pair.get("p_value"))
    q_value_change = _float_or_nan(loo_pair.get("q_value")) - _float_or_nan(baseline_pair.get("q_value"))
    return {
        "excluded_idx": str(excluded_idx),
        "mean_delta_change": mean_delta_change,
        "effect_size_change": effect_size_change,
        "p_value_change": p_value_change,
        "q_value_change": q_value_change,
        "abs_mean_delta_change": abs(mean_delta_change) if pd.notna(mean_delta_change) else 0.0,
        "abs_effect_size_change": abs(effect_size_change) if pd.notna(effect_size_change) else 0.0,
        "abs_p_value_change": abs(p_value_change) if pd.notna(p_value_change) else 0.0,
        "abs_q_value_change": abs(q_value_change) if pd.notna(q_value_change) else 0.0,
        "baseline_significant": bool(baseline_sig),
        "loo_significant": bool(loo_sig),
        "significance_switched": bool(baseline_sig != loo_sig),
    }


def _build_regression_change_record(
    baseline_regression: dict[str, Any],
    loo_regression: dict[str, Any],
    *,
    excluded_idx: str,
) -> dict[str, Any]:
    baseline_valid = not baseline_regression.get("message")
    loo_valid = not loo_regression.get("message")
    baseline_lookup = _coefficient_lookup(baseline_regression) if baseline_valid else {}
    loo_lookup = _coefficient_lookup(loo_regression) if loo_valid else {}
    all_names = sorted(set(baseline_lookup) | set(loo_lookup))
    coefficient_changes = []
    for name in all_names:
        baseline_estimate = _float_or_nan(baseline_lookup.get(name, {}).get("estimate"))
        loo_estimate = _float_or_nan(loo_lookup.get(name, {}).get("estimate"))
        change = loo_estimate - baseline_estimate
        coefficient_changes.append(
            {
                "name": name,
                "label": loo_lookup.get(name, baseline_lookup.get(name, {})).get("label", name),
                "change": change,
            }
        )
    baseline_primary = baseline_regression.get("primary_effect") or {}
    loo_primary = loo_regression.get("primary_effect") or {}
    primary_effect_change = _float_or_nan(loo_primary.get("estimate")) - _float_or_nan(baseline_primary.get("estimate"))
    return {
        "excluded_idx": str(excluded_idx),
        "model_valid": bool(baseline_valid and loo_valid),
        "primary_effect_label": loo_primary.get("label") or baseline_primary.get("label"),
        "primary_effect_change": primary_effect_change,
        "abs_primary_effect_change": abs(primary_effect_change) if pd.notna(primary_effect_change) else 0.0,
        "coefficient_changes": coefficient_changes,
        "r_squared_change": _float_or_nan(loo_regression.get("r_squared")) - _float_or_nan(baseline_regression.get("r_squared")),
    }


def _compute_correlation_summary(
    frame: pd.DataFrame,
    *,
    correlation_variable: str,
    correlation_method: str,
    group_label: str | None = None,
) -> dict[str, Any]:
    x = frame[correlation_variable].to_numpy(dtype=float)
    y = frame["delta"].to_numpy(dtype=float)
    if {"idx", "mtime"}.issubset(frame.columns):
        point_labels = (frame["idx"].astype(str) + "-" + frame["mtime"].astype(str)).tolist()
    elif "idx" in frame.columns:
        point_labels = frame["idx"].astype(str).tolist()
    else:
        point_labels = [str(index) for index in range(len(frame))]
    if correlation_method == "spearman":
        corr = stats.spearmanr(y, x)
        statistic = float(corr.statistic)
        p_value = float(corr.pvalue)
    else:
        statistic, p_value = stats.pearsonr(y, x)
        statistic = float(statistic)
        p_value = float(p_value)

    fit = stats.linregress(x, y)
    residuals = y - (fit.intercept + fit.slope * x)
    rmse = float(np.sqrt(np.sum(residuals**2) / max(len(x) - 2, 1)))
    result = {
        "variable": correlation_variable,
        "method": correlation_method,
        "n": int(len(frame)),
        "statistic": statistic,
        "p_value": p_value,
        "x": x.tolist(),
        "y": y.tolist(),
        "point_labels": point_labels,
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue**2),
        "slope_stderr": float(fit.stderr) if fit.stderr is not None else np.nan,
        "intercept_stderr": float(fit.intercept_stderr) if fit.intercept_stderr is not None else np.nan,
        "fit_p_value": float(fit.pvalue),
        "rmse": rmse,
    }
    if group_label is not None:
        result["group_label"] = group_label
    return result


def _compare_group_correlations(group_results: list[dict[str, Any]], correlation_method: str) -> dict[str, Any]:
    valid_groups = [group for group in group_results if "message" not in group and pd.notna(group["statistic"]) and group["n"] > 3]
    if len(valid_groups) != 2:
        return {
            "method": correlation_method,
            "message": "Group correlation comparison is available only for exactly two groups with at least 4 observations each.",
        }

    first, second = valid_groups
    z_first = np.arctanh(np.clip(first["statistic"], -0.999999, 0.999999))
    z_second = np.arctanh(np.clip(second["statistic"], -0.999999, 0.999999))
    standard_error = np.sqrt((1 / (first["n"] - 3)) + (1 / (second["n"] - 3)))
    z_statistic = float((z_first - z_second) / standard_error)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_statistic))))
    return {
        "method": correlation_method,
        "group_a": first["group_label"],
        "group_b": second["group_label"],
        "n_group_a": int(first["n"]),
        "n_group_b": int(second["n"]),
        "statistic_a": float(first["statistic"]),
        "statistic_b": float(second["statistic"]),
        "z_statistic": z_statistic,
        "p_value": p_value,
        "test": "Fisher z comparison" if correlation_method == "pearson" else "Fisher z comparison (approximate for Spearman)",
    }


def _trial_delta_results(filtered: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    comparison = filtered[filtered["trial_id"].isin([config.trial_a, config.trial_b])].copy()
    pivoted = (
        comparison.pivot_table(
            index=["subject_id", "idx", "base_subject_id", "group_label", "mtime", "roi_from", "roi_to"],
            columns="trial_id",
            values="value",
        )
        .dropna(subset=[config.trial_a, config.trial_b])
        .reset_index()
    )
    pivoted["delta"] = pivoted[config.trial_b] - pivoted[config.trial_a]
    pivoted = pivoted.rename(columns={config.trial_a: "trial_a_value", config.trial_b: "trial_b_value"})
    return _analyze_by_group_selection(
        pivoted,
        config,
        detail_columns=["subject_id", "idx", "base_subject_id", "group_label", "mtime", "trial_a_value", "trial_b_value", "delta"],
        mode="trial_delta",
    )


def _longitudinal_trial_delta_results(filtered: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not config.longitudinal_column:
        return pd.DataFrame()

    comparison = filtered[
        (filtered["trial_id"].isin([config.trial_a, config.trial_b]))
        & (filtered[config.longitudinal_column].isin([config.longitudinal_value_a, config.longitudinal_value_b]))
    ].copy()

    measurement_deltas = (
        comparison.pivot_table(
            index=["idx", "base_subject_id", "subject_id", "group_label", "roi_from", "roi_to", config.longitudinal_column],
            columns="trial_id",
            values="value",
        )
        .dropna(subset=[config.trial_a, config.trial_b])
        .reset_index()
    )
    measurement_deltas["trial_delta"] = measurement_deltas[config.trial_b] - measurement_deltas[config.trial_a]

    if not config.longitudinal_require_pairs:
        return _longitudinal_trial_delta_results_unpaired(measurement_deltas, config)

    paired = (
        measurement_deltas.pivot_table(
            index=["idx", "base_subject_id", "group_label", "roi_from", "roi_to"],
            columns=config.longitudinal_column,
            values="trial_delta",
        )
        .dropna(subset=[config.longitudinal_value_a, config.longitudinal_value_b])
        .reset_index()
    )
    paired["delta"] = paired[config.longitudinal_value_b] - paired[config.longitudinal_value_a]

    detail_lookup = measurement_deltas[
        ["idx", "subject_id", "roi_from", "roi_to", config.longitudinal_column, config.trial_a, config.trial_b, "trial_delta"]
    ].drop_duplicates()

    detail_records = []
    for _, row in paired.iterrows():
        pair_subjects = detail_lookup[
            (detail_lookup["idx"] == row["idx"])
            & (detail_lookup["roi_from"] == row["roi_from"])
            & (detail_lookup["roi_to"] == row["roi_to"])
        ]
        record_a = pair_subjects[pair_subjects[config.longitudinal_column] == config.longitudinal_value_a].head(1)
        record_b = pair_subjects[pair_subjects[config.longitudinal_column] == config.longitudinal_value_b].head(1)
        detail_records.append(
            {
                "base_subject_id": row["idx"],
                "idx": row["idx"],
                "group_label": row["group_label"],
                "roi_from": row["roi_from"],
                "roi_to": row["roi_to"],
                "subject_id_a": record_a["subject_id"].iloc[0] if not record_a.empty else None,
                "subject_id_b": record_b["subject_id"].iloc[0] if not record_b.empty else None,
                "measure_a_value": float(row[config.longitudinal_value_a]),
                "measure_b_value": float(row[config.longitudinal_value_b]),
                "delta_a": float(row[config.longitudinal_value_a]),
                "delta_b": float(row[config.longitudinal_value_b]),
                "delta": float(row["delta"]),
            }
        )
    detail_frame = pd.DataFrame(detail_records)
    return _analyze_by_group_selection(
        detail_frame,
        config,
        detail_columns=["base_subject_id", "idx", "group_label", "subject_id_a", "subject_id_b", "measure_a_value", "measure_b_value", "delta_a", "delta_b", "delta"],
        mode="longitudinal_delta",
    )


def _longitudinal_trial_delta_results_unpaired(measurement_deltas: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_a = str(config.group_a) if config.group_a is not None else None
    group_b = str(config.group_b) if config.group_b is not None else None
    measurement_deltas = measurement_deltas.copy()
    measurement_deltas["group_label"] = measurement_deltas["group_label"].astype(str)
    measurement_deltas["timepoint_label"] = measurement_deltas[config.longitudinal_column].astype(str)

    pooled_all_groups = group_a == ALL_GROUPS_VALUE or group_b == ALL_GROUPS_VALUE
    if pooled_all_groups:
        subset = measurement_deltas
        results = _unpaired_timepoint_results(
            subset,
            config,
            plot_group_builder=lambda row: str(row["timepoint_label"]),
            summary_label_builder=lambda row: str(row["timepoint_label"]),
        )
        context = {"group_a": ALL_GROUPS_LABEL, "group_b": ALL_GROUPS_LABEL, "between_groups": False}
    elif group_a == group_b or group_b is None:
        target_group = group_a
        subset = measurement_deltas[measurement_deltas["group_label"] == target_group]
        results = _unpaired_timepoint_results(
            subset,
            config,
            plot_group_builder=lambda row: str(row["timepoint_label"]),
            summary_label_builder=lambda row: str(row["timepoint_label"]),
        )
        context = {"group_a": target_group, "group_b": target_group, "between_groups": False}
    else:
        subset = measurement_deltas[measurement_deltas["group_label"].isin([group_a, group_b])]
        results = _unpaired_group_timepoint_interaction_results(subset, config, group_a, group_b)
        context = {"group_a": group_a, "group_b": group_b, "between_groups": True}
    return _attach_q_values(results, config.correction_method), context


def _unpaired_timepoint_results(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    plot_group_builder,
    summary_label_builder,
) -> pd.DataFrame:
    results = []
    time_a = str(config.longitudinal_value_a)
    time_b = str(config.longitudinal_value_b)
    for (roi_from, roi_to), group in frame.groupby(["roi_from", "roi_to"], sort=True):
        first = group[group["timepoint_label"] == time_a].copy()
        second = group[group["timepoint_label"] == time_b].copy()
        delta_a = first["trial_delta"].to_numpy(dtype=float)
        delta_b = second["trial_delta"].to_numpy(dtype=float)
        n_a = len(delta_a)
        n_b = len(delta_b)
        if n_a >= 2 and n_b >= 2:
            statistic, p_value = stats.ttest_ind(delta_b, delta_a, equal_var=False)
            pooled_var = (((n_a - 1) * np.var(delta_a, ddof=1)) + ((n_b - 1) * np.var(delta_b, ddof=1))) / max(n_a + n_b - 2, 1)
            pooled_sd = float(np.sqrt(pooled_var)) if pooled_var >= 0 else np.nan
            effect_size = float((np.mean(delta_b) - np.mean(delta_a)) / pooled_sd) if pooled_sd and pooled_sd > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan

        details = []
        for subgroup, label in ((first, time_a), (second, time_b)):
            subgroup = subgroup.copy()
            subgroup["delta"] = subgroup["trial_delta"].astype(float)
            subgroup["plot_group_label"] = subgroup.apply(plot_group_builder, axis=1)
            subgroup["summary_group_label"] = subgroup.apply(summary_label_builder, axis=1)
            details.extend(
                subgroup[
                    ["base_subject_id", "idx", "subject_id", "group_label", "timepoint_label", "plot_group_label", "summary_group_label", "delta"]
                ].to_dict(orient="records")
            )

        results.append(
            {
                "pair_key": f"{roi_from}|{roi_to}",
                "roi_from": roi_from,
                "roi_to": roi_to,
                "mean_delta": float(np.mean(delta_b) - np.mean(delta_a)) if n_a and n_b else np.nan,
                "effect_size": effect_size,
                "statistic": float(statistic) if pd.notna(statistic) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "n": int(n_a + n_b),
                "n_group_a": int(n_a),
                "n_group_b": int(n_b),
                "group_stats": [
                    _group_summary(time_a, delta_a),
                    _group_summary(time_b, delta_b),
                ],
                "detail_records": details,
            }
        )
    return pd.DataFrame(results)


def _unpaired_group_timepoint_interaction_results(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    results = []
    time_a = str(config.longitudinal_value_a)
    time_b = str(config.longitudinal_value_b)
    for (roi_from, roi_to), group in frame.groupby(["roi_from", "roi_to"], sort=True):
        group = group.copy()
        group["time_indicator"] = (group["timepoint_label"] == time_b).astype(int)
        group["group_indicator"] = (group["group_label"] == group_b).astype(int)
        group["interaction"] = group["time_indicator"] * group["group_indicator"]

        subgroup_stats = []
        detail_records = []
        cell_means: dict[tuple[str, str], float] = {}
        for current_group in (group_a, group_b):
            for current_time in (time_a, time_b):
                cell = group[(group["group_label"] == current_group) & (group["timepoint_label"] == current_time)].copy()
                values = cell["trial_delta"].to_numpy(dtype=float)
                subgroup_stats.append(_group_summary(f"{current_group} | {current_time}", values))
                cell_means[(current_group, current_time)] = float(np.mean(values)) if len(values) else np.nan
                if not cell.empty:
                    cell["delta"] = cell["trial_delta"].astype(float)
                    cell["plot_group_label"] = cell["group_label"].astype(str) + " | " + cell["timepoint_label"].astype(str)
                    cell["summary_group_label"] = cell["plot_group_label"]
                    detail_records.extend(
                        cell[
                            ["base_subject_id", "idx", "subject_id", "group_label", "timepoint_label", "plot_group_label", "summary_group_label", "delta"]
                        ].to_dict(orient="records")
                    )

        mean_delta = (
            cell_means[(group_b, time_b)]
            - cell_means[(group_b, time_a)]
            - cell_means[(group_a, time_b)]
            + cell_means[(group_a, time_a)]
        )

        if len(group) >= 4 and group["interaction"].nunique() > 1 and group["time_indicator"].nunique() > 1 and group["group_indicator"].nunique() > 1:
            predictors = sm.add_constant(group[["time_indicator", "group_indicator", "interaction"]], has_constant="add")
            model = sm.OLS(group["trial_delta"], predictors).fit()
            statistic = float(model.tvalues["interaction"])
            p_value = float(model.pvalues["interaction"])
            residual_sd = float(np.sqrt(model.mse_resid)) if model.df_resid > 0 else np.nan
            effect_size = float(mean_delta / residual_sd) if residual_sd and residual_sd > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan

        results.append(
            {
                "pair_key": f"{roi_from}|{roi_to}",
                "roi_from": roi_from,
                "roi_to": roi_to,
                "mean_delta": float(mean_delta) if pd.notna(mean_delta) else np.nan,
                "effect_size": effect_size,
                "statistic": statistic,
                "p_value": p_value,
                "n": int(len(group)),
                "n_group_a": int(len(group[group["group_label"] == group_a])),
                "n_group_b": int(len(group[group["group_label"] == group_b])),
                "group_stats": subgroup_stats,
                "detail_records": detail_records,
            }
        )
    return pd.DataFrame(results)


def _analyze_by_group_selection(frame: pd.DataFrame, config: AnalysisConfig, detail_columns: list[str], mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_a = str(config.group_a) if config.group_a is not None else None
    group_b = str(config.group_b) if config.group_b is not None else None
    if group_a is None and not frame.empty:
        group_a = str(frame["group_label"].dropna().iloc[0])
    if group_b is None:
        group_b = group_a

    frame = frame.copy()
    frame["group_label"] = frame["group_label"].astype(str)

    pooled_all_groups = group_a == ALL_GROUPS_VALUE or group_b == ALL_GROUPS_VALUE
    if pooled_all_groups:
        results = _within_group_results(frame, detail_columns, mode, group_name_override=ALL_GROUPS_LABEL)
        context = {"group_a": ALL_GROUPS_LABEL, "group_b": ALL_GROUPS_LABEL, "between_groups": False}
    elif group_a == group_b:
        subset = frame[frame["group_label"] == group_a] if group_a is not None else frame
        results = _within_group_results(subset, detail_columns, mode, group_name_override=group_a)
        context = {"group_a": group_a, "group_b": group_b, "between_groups": False}
    else:
        subset = frame[frame["group_label"].isin([group_a, group_b])]
        results = _between_group_results(subset, detail_columns, group_a, group_b)
        context = {"group_a": group_a, "group_b": group_b, "between_groups": True}
    return _attach_q_values(results, config.correction_method), context


def _within_group_results(frame: pd.DataFrame, detail_columns: list[str], mode: str, group_name_override: str | None = None) -> pd.DataFrame:
    results = []
    for (roi_from, roi_to), group in frame.groupby(["roi_from", "roi_to"], sort=True):
        deltas = group["delta"].to_numpy(dtype=float)
        n_obs = len(group)
        group_name = group_name_override or (str(group["group_label"].iloc[0]) if n_obs else "Group 1")
        if mode == "trial_delta":
            a_values = group["trial_a_value"].to_numpy(dtype=float)
            b_values = group["trial_b_value"].to_numpy(dtype=float)
        else:
            a_values = group["delta_a"].to_numpy(dtype=float)
            b_values = group["delta_b"].to_numpy(dtype=float)
        if n_obs >= 2:
            statistic, p_value = stats.ttest_rel(b_values, a_values)
            std_delta = np.std(deltas, ddof=1)
            effect_size = float(np.mean(deltas) / std_delta) if std_delta > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan
        results.append(
            {
                "pair_key": f"{roi_from}|{roi_to}",
                "roi_from": roi_from,
                "roi_to": roi_to,
                "mean_delta": float(np.mean(deltas)) if n_obs else np.nan,
                "effect_size": effect_size,
                "statistic": float(statistic) if pd.notna(statistic) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "n": int(n_obs),
                "n_group_a": int(n_obs),
                "n_group_b": int(n_obs),
                "group_stats": [
                    _group_summary(group_name, deltas),
                ],
                "detail_records": group[detail_columns].to_dict(orient="records"),
            }
        )
    return pd.DataFrame(results)


def _between_group_results(frame: pd.DataFrame, detail_columns: list[str], group_a: str, group_b: str) -> pd.DataFrame:
    results = []
    for (roi_from, roi_to), group in frame.groupby(["roi_from", "roi_to"], sort=True):
        first = group[group["group_label"] == group_a]
        second = group[group["group_label"] == group_b]
        delta_a = first["delta"].to_numpy(dtype=float)
        delta_b = second["delta"].to_numpy(dtype=float)
        n_a = len(delta_a)
        n_b = len(delta_b)
        if n_a >= 2 and n_b >= 2:
            statistic, p_value = stats.ttest_ind(delta_b, delta_a, equal_var=False)
            pooled_var = (((n_a - 1) * np.var(delta_a, ddof=1)) + ((n_b - 1) * np.var(delta_b, ddof=1))) / max(n_a + n_b - 2, 1)
            pooled_sd = float(np.sqrt(pooled_var)) if pooled_var >= 0 else np.nan
            effect_size = float((np.mean(delta_b) - np.mean(delta_a)) / pooled_sd) if pooled_sd and pooled_sd > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan
        results.append(
            {
                "pair_key": f"{roi_from}|{roi_to}",
                "roi_from": roi_from,
                "roi_to": roi_to,
                "mean_delta": float(np.mean(delta_b) - np.mean(delta_a)) if n_a and n_b else np.nan,
                "effect_size": effect_size,
                "statistic": float(statistic) if pd.notna(statistic) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "n": int(n_a + n_b),
                "n_group_a": int(n_a),
                "n_group_b": int(n_b),
                "group_stats": [
                    _group_summary(group_a, delta_a),
                    _group_summary(group_b, delta_b),
                ],
                "detail_records": group[detail_columns].to_dict(orient="records"),
            }
        )
    return pd.DataFrame(results)


def _attach_q_values(results: pd.DataFrame, correction_method: str) -> pd.DataFrame:
    if results.empty:
        results["q_value"] = []
        return results
    results = results.copy()
    mask = results["p_value"].notna()
    mask_indices = np.where(mask)[0]

    for method in CORRECTION_METHODS:
        corrected_values = np.full(len(results), np.nan)
        if mask.any():
            if method == "none":
                corrected_values[mask_indices] = results.loc[mask, "p_value"].to_numpy()
            else:
                _, corrected, _, _ = multipletests(results.loc[mask, "p_value"], method=method)
                corrected_values[mask_indices] = corrected
        results[f"q_value_{method}"] = corrected_values

    results["q_value"] = results[f"q_value_{correction_method}"]
    return results


def _group_summary(group_name: str, deltas: np.ndarray) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=float)
    n_obs = len(deltas)
    if n_obs == 0:
        return {
            "group_label": group_name,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
        }

    std = float(np.std(deltas, ddof=1)) if n_obs >= 2 else np.nan
    sem = float(std / np.sqrt(n_obs)) if n_obs >= 2 and pd.notna(std) else np.nan
    return {
        "group_label": group_name,
        "n": int(n_obs),
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "std": std,
        "sem": sem,
    }


def _correction_label(correction_method: str) -> str:
    labels = {
        "none": "None (raw p-values)",
        "fdr_bh": "FDR-BH",
        "bonferroni": "Bonferroni",
        "holm": "Holm",
    }
    return labels.get(correction_method, correction_method)


def _build_matrix(results: pd.DataFrame, roi_order: list[str], value_column: str) -> np.ndarray:
    matrix = np.full((len(roi_order), len(roi_order)), np.nan)
    lookup = {name: index for index, name in enumerate(roi_order)}
    for _, row in results.iterrows():
        i = lookup[row["roi_from"]]
        j = lookup[row["roi_to"]]
        matrix[i, j] = row[value_column]
        matrix[j, i] = row[value_column]
    return matrix


def derive_roi_metadata(roi_names: list[str]) -> list[dict[str, Any]]:
    """Infer hemisphere, anatomical class, and homologues from ROI labels."""
    metadata: list[dict[str, Any]] = []
    name_lookup = {name: name for name in roi_names}
    for roi in roi_names:
        lower = roi.lower()
        hemisphere = "midline"
        if lower.endswith("_left") or lower.endswith("_l"):
            hemisphere = "left"
        elif lower.endswith("_right") or lower.endswith("_r"):
            hemisphere = "right"

        anatomical_class = "other"
        if "fronto-central" in lower or "frontocentral" in lower:
            anatomical_class = "fronto-central"
        elif "frontal" in lower:
            anatomical_class = "frontal"
        elif "temporal" in lower:
            anatomical_class = "temporal"
        elif "parietooccipital" in lower:
            anatomical_class = "parietooccipital"
        elif "parietal" in lower:
            anatomical_class = "parietal"
        elif "occipital" in lower:
            anatomical_class = "occipital"

        homologue = None
        if hemisphere == "left":
            candidate = roi.replace("_left", "_right").replace("_l", "_r")
            homologue = name_lookup.get(candidate)
        elif hemisphere == "right":
            candidate = roi.replace("_right", "_left").replace("_r", "_l")
            homologue = name_lookup.get(candidate)

        metadata.append(
            {
                "roi": roi,
                "hemisphere": hemisphere,
                "anatomical_class": anatomical_class,
                "homologue": homologue,
            }
        )
    return metadata


def run_network_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    config: NetworkAnalysisConfig,
) -> dict[str, Any]:
    """Compute network-level summary, graph, community, and NBS results."""
    detail_frame = _build_network_detail_frame(analysis_result)
    roi_metadata = derive_roi_metadata(list(bundle.channels))
    metadata_frame = pd.DataFrame(roi_metadata)
    observation_frame, observation_matrices = _build_observation_matrices(detail_frame, list(bundle.channels))

    summary_records = _build_network_summary_scores(observation_matrices, metadata_frame)
    if not summary_records.empty and not observation_frame.empty:
        summary_records = summary_records.merge(observation_frame, on="observation_id", how="left", validate="many_to_one")
    summary_results = _analyze_network_scores(summary_records, analysis_result)

    graph_data = _compute_graph_metrics_for_observations(
        observation_frame=observation_frame,
        observation_matrices=observation_matrices,
        roi_order=list(bundle.channels),
        roi_metadata=metadata_frame,
        config=config,
    )
    graph_global_results = _analyze_network_scores(graph_data["global_records"], analysis_result)
    graph_node_results = _analyze_network_scores(graph_data["node_records"], analysis_result)

    community_results = _compute_community_results(
        observation_frame=observation_frame,
        observation_matrices=observation_matrices,
        roi_order=list(bundle.channels),
        roi_metadata=metadata_frame,
        config=config,
        analysis_result=analysis_result,
    )
    nbs_results = _compute_nbs_results(
        observation_frame=observation_frame,
        observation_matrices=observation_matrices,
        roi_order=list(bundle.channels),
        config=config,
        analysis_result=analysis_result,
    )

    return {
        "mode": config.mode,
        "roi_metadata": roi_metadata,
        "observation_count": int(len(observation_frame)),
        "network_summary_results": summary_results,
        "graph_metric_results": {
            "global": graph_global_results,
            "node": graph_node_results,
            "raw_global_records": graph_data["global_records"].to_dict(orient="records"),
            "raw_node_records": graph_data["node_records"].to_dict(orient="records"),
        },
        "community_results": community_results,
        "nbs_results": nbs_results,
    }


def run_patterns_analysis(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    config: PatternsAnalysisConfig,
) -> dict[str, Any]:
    """Compute exploratory embedding, clustering, feature-pattern, and brain-behavior summaries."""
    feature_data = build_patterns_feature_data(bundle, analysis_result, config)
    metadata = feature_data["metadata"]
    if metadata.empty or feature_data["matrix"].size == 0:
        message = "No complete subject-level pattern features are available for the current selection."
        return {
            "message": message,
            "feature_matrix": feature_data,
            "embedding_results": {"message": message},
            "subject_cluster_results": {"message": message},
            "feature_pattern_results": {"message": message},
            "brain_behavior_results": {"message": message},
        }

    embedding_results = _compute_patterns_embedding(feature_data, config)
    subject_cluster_results = _compute_patterns_subject_clustering(feature_data, config, embedding_results)
    feature_pattern_results = _compute_patterns_feature_patterns(feature_data, config)
    brain_behavior_results = _compute_patterns_brain_behavior(bundle, feature_data, config)
    return {
        "message": None,
        "feature_matrix": feature_data,
        "embedding_results": embedding_results,
        "subject_cluster_results": subject_cluster_results,
        "feature_pattern_results": feature_pattern_results,
        "brain_behavior_results": brain_behavior_results,
    }


def build_patterns_feature_data(
    bundle: DatasetBundle,
    analysis_result: dict[str, Any],
    config: PatternsAnalysisConfig,
) -> dict[str, Any]:
    """Build the observation x feature matrix used by the Patterns panel."""
    detail_frame = _build_network_detail_frame(analysis_result)
    observation_frame, observation_matrices = _build_observation_matrices(detail_frame, list(bundle.channels))
    if observation_frame.empty or not observation_matrices:
        return {
            "matrix": np.empty((0, 0), dtype=float),
            "matrix_scaled": np.empty((0, 0), dtype=float),
            "metadata": pd.DataFrame(),
            "feature_labels": [],
            "roi_order": list(bundle.channels),
        }

    roi_order = list(bundle.channels)
    feature_labels = [f"{roi_order[i]}|{roi_order[j]}" for i in range(len(roi_order)) for j in range(i + 1, len(roi_order))]
    rows: list[np.ndarray] = []
    valid_ids: list[str] = []
    for observation_id in observation_frame["observation_id"].astype(str).tolist():
        matrix = observation_matrices.get(observation_id)
        if matrix is None:
            continue
        row = np.asarray([matrix[i, j] for i in range(len(roi_order)) for j in range(i + 1, len(roi_order))], dtype=float)
        rows.append(row)
        valid_ids.append(observation_id)
    matrix = np.vstack(rows) if rows else np.empty((0, len(feature_labels)), dtype=float)
    metadata = observation_frame[observation_frame["observation_id"].astype(str).isin(valid_ids)].copy().reset_index(drop=True)
    metadata = _attach_patterns_subject_metadata(bundle, metadata)

    if config.missing_data_mode == "complete_cases" and matrix.size:
        mask = np.isfinite(matrix).all(axis=1)
        matrix = matrix[mask]
        metadata = metadata.loc[mask].reset_index(drop=True)

    matrix_scaled, feature_means, feature_stds = _standardize_patterns_matrix(matrix, config.standardize)
    return {
        "matrix": matrix,
        "matrix_scaled": matrix_scaled,
        "metadata": metadata,
        "feature_labels": feature_labels,
        "roi_order": roi_order,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
    }


def _build_network_detail_frame(analysis_result: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in analysis_result.get("pair_results", []):
        for detail in record.get("detail_records", []):
            row = dict(detail)
            row["roi_from"] = record["roi_from"]
            row["roi_to"] = record["roi_to"]
            rows.append(row)
    return pd.DataFrame(rows)


def _attach_patterns_subject_metadata(bundle: DatasetBundle, observation_frame: pd.DataFrame) -> pd.DataFrame:
    if observation_frame.empty:
        return observation_frame
    frame = observation_frame.copy()
    if "subject_id" in frame.columns:
        subject_columns = [column for column in bundle.subjects.columns if column not in frame.columns or column == "subject_id"]
        return frame.merge(bundle.subjects[subject_columns], on="subject_id", how="left", validate="many_to_one")
    if "base_subject_id" in frame.columns:
        baseline = (
            bundle.subjects.assign(mtime_order=bundle.subjects["mtime"].str.extract(r"(\d+)").fillna("999").astype(int))
            .sort_values(["base_subject_id", "mtime_order"])
            .drop_duplicates("base_subject_id", keep="first")
            .drop(columns="mtime_order")
        )
        baseline_columns = [column for column in baseline.columns if column not in frame.columns or column == "base_subject_id"]
        return frame.merge(baseline[baseline_columns], on="base_subject_id", how="left", validate="many_to_one")
    return frame


def _standardize_patterns_matrix(matrix: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return matrix, np.array([], dtype=float), np.array([], dtype=float)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    if mode == "zscore":
        scaled = (matrix - means) / stds
    else:
        scaled = matrix.copy()
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled, means, stds


def _build_observation_matrices(detail_frame: pd.DataFrame, roi_order: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if detail_frame.empty:
        return pd.DataFrame(), {}

    frame = detail_frame.copy()
    if {"delta_a", "delta_b", "base_subject_id"}.issubset(frame.columns):
        frame["observation_id"] = frame["base_subject_id"].astype(str)
        frame["observation_label"] = frame["base_subject_id"].astype(str)
        frame["analysis_variant"] = "paired_longitudinal"
    elif "timepoint_label" in frame.columns:
        base = frame["idx"].astype(str) if "idx" in frame.columns else frame["subject_id"].astype(str)
        frame["observation_id"] = base + "|" + frame["timepoint_label"].astype(str)
        frame["observation_label"] = frame["observation_id"]
        frame["analysis_variant"] = "timepoint_level"
    else:
        base = frame["idx"].astype(str) if "idx" in frame.columns else frame["subject_id"].astype(str)
        frame["observation_id"] = base
        frame["observation_label"] = base
        frame["analysis_variant"] = "subject_level"

    metadata_columns = ["observation_id", "observation_label", "group_label", "analysis_variant"]
    for optional in ("idx", "subject_id", "base_subject_id", "timepoint_label", "mtime"):
        if optional in frame.columns:
            metadata_columns.append(optional)
    observation_frame = frame[metadata_columns].drop_duplicates("observation_id").reset_index(drop=True)

    matrices: dict[str, np.ndarray] = {}
    lookup = {name: index for index, name in enumerate(roi_order)}
    for observation_id, group in frame.groupby("observation_id", sort=True):
        matrix = np.full((len(roi_order), len(roi_order)), np.nan)
        for _, row in group.iterrows():
            i = lookup[str(row["roi_from"])]
            j = lookup[str(row["roi_to"])]
            matrix[i, j] = float(row["delta"])
            matrix[j, i] = float(row["delta"])
        np.fill_diagonal(matrix, 0.0)
        matrices[str(observation_id)] = matrix
    return observation_frame, matrices


def _compute_patterns_embedding(feature_data: dict[str, Any], config: PatternsAnalysisConfig) -> dict[str, Any]:
    matrix = feature_data["matrix_scaled"]
    metadata = feature_data["metadata"].copy()
    if matrix.size == 0 or metadata.empty:
        return {"message": "No embedding could be computed."}
    method = (config.embedding_method or "pca").lower()
    if method != "pca":
        return {
            "message": f"Embedding method '{config.embedding_method}' is not available in the current environment. PCA is supported without extra dependencies.",
            "method": method,
        }
    dim = min(max(int(config.embedding_dim), 2), 3, matrix.shape[0], matrix.shape[1] if matrix.shape[1] else 2)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :dim] * s[:dim]
    total_var = np.sum(s**2)
    explained = ((s[:dim] ** 2) / total_var).tolist() if total_var > 0 else [np.nan] * dim
    loadings = vt[:dim, :]
    metadata = metadata.copy()
    for index in range(dim):
        metadata[f"component_{index + 1}"] = scores[:, index]
    top_features = []
    for component_index in range(dim):
        weights = pd.Series(loadings[component_index], index=feature_data["feature_labels"], dtype=float)
        strongest = weights.abs().sort_values(ascending=False).head(8)
        top_features.append(
            {
                "component": component_index + 1,
                "features": [
                    {"feature": str(feature), "loading": float(weights.loc[feature])}
                    for feature in strongest.index.tolist()
                ],
            }
        )
    return {
        "message": None,
        "method": "pca",
        "dim": dim,
        "metadata": metadata.to_dict(orient="records"),
        "explained_variance_ratio": explained,
        "top_feature_loadings": top_features,
    }


def _compute_patterns_subject_clustering(
    feature_data: dict[str, Any],
    config: PatternsAnalysisConfig,
    embedding_results: dict[str, Any],
) -> dict[str, Any]:
    matrix = feature_data["matrix_scaled"]
    metadata = feature_data["metadata"].copy()
    if matrix.size == 0 or metadata.empty:
        return {"message": "No subject clustering is available."}
    method = (config.cluster_method or "kmeans").lower()
    n_clusters = max(2, min(int(config.n_clusters), len(metadata)))
    labels: np.ndarray
    if method == "kmeans":
        _, labels = kmeans2(matrix, n_clusters, minit="points")
        labels = np.asarray(labels, dtype=int)
    elif method == "hierarchical":
        linkage_matrix = linkage(matrix, method="ward")
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust").astype(int) - 1
    else:
        return {
            "message": f"Cluster method '{config.cluster_method}' is not available without extra dependencies. Supported methods are k-means and hierarchical clustering.",
            "method": method,
        }

    metadata = metadata.copy()
    metadata["cluster_label"] = labels
    cluster_rows = []
    numeric_columns = [column for column in metadata.columns if str(metadata[column].dtype) != "object" and column != "cluster_label"]
    for cluster_id, group in metadata.groupby("cluster_label", sort=True):
        row: dict[str, Any] = {
            "cluster_label": int(cluster_id),
            "n": int(len(group)),
            "groups": ", ".join(sorted(group["group_label"].dropna().astype(str).unique().tolist())) if "group_label" in group.columns else "n/a",
            "timepoints": ", ".join(sorted(group["timepoint_label"].dropna().astype(str).unique().tolist())) if "timepoint_label" in group.columns else "n/a",
        }
        for column in numeric_columns[:4]:
            row[f"mean_{column}"] = float(np.nanmean(group[column].to_numpy(dtype=float))) if len(group[column].dropna()) else np.nan
        cluster_rows.append(row)
    return {
        "message": None,
        "method": method,
        "n_clusters": n_clusters,
        "metadata": metadata.to_dict(orient="records"),
        "cluster_summary": cluster_rows,
        "embedding_overlay": embedding_results.get("metadata", []),
    }


def _compute_patterns_feature_patterns(feature_data: dict[str, Any], config: PatternsAnalysisConfig) -> dict[str, Any]:
    matrix = feature_data["matrix_scaled"]
    metadata = feature_data["metadata"]
    if matrix.size == 0 or metadata.empty:
        return {"message": "No feature-pattern analysis is available."}
    level = (config.feature_pattern_level or "edges").lower()
    if level == "edges":
        pattern_matrix = matrix.T
        labels = feature_data["feature_labels"]
    else:
        roi_order = feature_data["roi_order"]
        pattern_rows = []
        for roi in roi_order:
            indices = [index for index, label in enumerate(feature_data["feature_labels"]) if roi in label.split("|")]
            pattern_rows.append(np.nanmean(matrix[:, indices], axis=1) if indices else np.zeros(matrix.shape[0], dtype=float))
        pattern_matrix = np.vstack(pattern_rows)
        labels = roi_order
    if pattern_matrix.shape[0] < 2:
        return {"message": "Not enough features are available for clustering patterns."}
    correlation = np.corrcoef(pattern_matrix)
    correlation = np.nan_to_num(correlation, nan=0.0)
    distance = pdist(pattern_matrix, metric="correlation")
    if np.isnan(distance).all() or len(distance) == 0:
        order = np.arange(pattern_matrix.shape[0], dtype=int)
        linkage_matrix = np.empty((0, 4), dtype=float)
    else:
        distance = np.nan_to_num(distance, nan=1.0)
        linkage_matrix = linkage(distance, method="average")
        order = leaves_list(linkage_matrix)
    ordered_labels = [labels[index] for index in order.tolist()]
    ordered_corr = correlation[np.ix_(order, order)]
    return {
        "message": None,
        "level": level,
        "labels": labels,
        "ordered_labels": ordered_labels,
        "correlation_matrix": correlation.tolist(),
        "ordered_correlation_matrix": ordered_corr.tolist(),
    }


def _compute_patterns_brain_behavior(
    bundle: DatasetBundle,
    feature_data: dict[str, Any],
    config: PatternsAnalysisConfig,
) -> dict[str, Any]:
    metadata = feature_data["metadata"].copy()
    matrix = feature_data["matrix_scaled"]
    behavior_variables = list(config.behavior_variables or [])
    if not behavior_variables:
        numeric_candidates = [
            column
            for column in metadata.columns
            if column not in {"dataset_id", "subject_id", "base_subject_id", "group_label", "mtime", "timepoint_label", "observation_id", "observation_label", "analysis_variant"}
            and str(metadata[column].dtype) != "object"
        ]
        behavior_variables = numeric_candidates[:3]
    if not behavior_variables:
        return {"message": "No numeric behavior variables are available for multivariate pattern analysis."}
    available = [column for column in behavior_variables if column in metadata.columns]
    if not available:
        return {"message": "The selected behavior variables are not available for the current observations."}
    behavior = metadata[available].apply(pd.to_numeric, errors="coerce")
    mask = behavior.notna().all(axis=1).to_numpy(dtype=bool)
    if mask.sum() < max(3, min(config.n_components, len(available)) + 1):
        return {"message": "Not enough complete observations are available for the selected behavior variables."}
    x = matrix[mask]
    y = behavior.loc[mask].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    y_mean = np.mean(y, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, ddof=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    y_scaled = (y - y_mean) / y_std

    method = (config.cca_pls_method or "pls").lower()
    if method != "pls":
        return {
            "message": f"Method '{config.cca_pls_method}' is not available in this dependency-light build. PLS is supported.",
            "method": method,
        }

    n_components = max(1, min(int(config.n_components), x.shape[0], x.shape[1], y_scaled.shape[1]))
    cross_cov = x.T @ y_scaled / max(len(x) - 1, 1)
    u, s, vt = np.linalg.svd(cross_cov, full_matrices=False)
    brain_weights = u[:, :n_components]
    behavior_weights = vt.T[:, :n_components]
    brain_scores = x @ brain_weights
    behavior_scores = y_scaled @ behavior_weights
    matched = metadata.loc[mask].copy().reset_index(drop=True)
    for index in range(n_components):
        matched[f"brain_score_{index + 1}"] = brain_scores[:, index]
        matched[f"behavior_score_{index + 1}"] = behavior_scores[:, index]

    feature_weights = []
    for component_index in range(n_components):
        series = pd.Series(brain_weights[:, component_index], index=feature_data["feature_labels"], dtype=float)
        top = series.abs().sort_values(ascending=False).head(8)
        feature_weights.append(
            {
                "component": component_index + 1,
                "features": [{"feature": str(label), "weight": float(series.loc[label])} for label in top.index.tolist()],
            }
        )
    behavior_weights_rows = []
    for component_index in range(n_components):
        behavior_weights_rows.append(
            {
                "component": component_index + 1,
                "variables": [{"variable": column, "weight": float(behavior_weights[idx, component_index])} for idx, column in enumerate(available)],
            }
        )
    singular_values = s[:n_components].tolist()
    return {
        "message": None,
        "method": "pls",
        "n_components": n_components,
        "behavior_variables": available,
        "scores": matched.to_dict(orient="records"),
        "singular_values": singular_values,
        "brain_feature_weights": feature_weights,
        "behavior_weights": behavior_weights_rows,
    }


def _build_network_summary_scores(observation_matrices: dict[str, np.ndarray], roi_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if roi_metadata.empty:
        return pd.DataFrame()

    roi_order = roi_metadata["roi"].tolist()
    hemispheres = roi_metadata.set_index("roi")["hemisphere"].to_dict()
    classes = roi_metadata.set_index("roi")["anatomical_class"].to_dict()
    homologues = roi_metadata.set_index("roi")["homologue"].to_dict()

    for observation_id, matrix in observation_matrices.items():
        for index, roi in enumerate(roi_order):
            weights = matrix[index, :]
            valid = np.isfinite(weights)
            valid[index] = False
            rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": f"roi_mean::{roi}",
                    "score_label": f"{roi} mean connectivity",
                    "score_family": "roi_mean_connectivity",
                    "delta": _safe_mean(weights[valid]),
                }
            )

            hemi = hemispheres.get(roi, "midline")
            if hemi in {"left", "right"}:
                same = np.array([hemispheres.get(other) == hemi and other != roi for other in roi_order], dtype=bool)
                opposite = np.array(
                    [
                        hemispheres.get(other) in {"left", "right"}
                        and hemispheres.get(other) != hemi
                        and other != roi
                        for other in roi_order
                    ],
                    dtype=bool,
                )
                rows.append(
                    {
                        "observation_id": observation_id,
                        "score_key": f"roi_ipsilateral::{roi}",
                        "score_label": f"{roi} ipsilateral mean",
                        "score_family": "roi_ipsilateral_connectivity",
                        "delta": _safe_mean(weights[same]),
                    }
                )
                rows.append(
                    {
                        "observation_id": observation_id,
                        "score_key": f"roi_contralateral::{roi}",
                        "score_label": f"{roi} contralateral mean",
                        "score_family": "roi_contralateral_connectivity",
                        "delta": _safe_mean(weights[opposite]),
                    }
                )

            same_class = np.array([classes.get(other) == classes.get(roi) and other != roi for other in roi_order], dtype=bool)
            other_class = np.array([classes.get(other) != classes.get(roi) and other != roi for other in roi_order], dtype=bool)
            rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": f"roi_within_class::{roi}",
                    "score_label": f"{roi} within-class mean",
                    "score_family": "roi_within_class_connectivity",
                    "delta": _safe_mean(weights[same_class]),
                }
            )
            rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": f"roi_between_class::{roi}",
                    "score_label": f"{roi} between-class mean",
                    "score_family": "roi_between_class_connectivity",
                    "delta": _safe_mean(weights[other_class]),
                }
            )

        class_groups = sorted(roi_metadata["anatomical_class"].dropna().astype(str).unique().tolist())
        for anatomical_class in class_groups:
            class_indices = [i for i, roi in enumerate(roi_order) if classes.get(roi) == anatomical_class]
            if len(class_indices) >= 2:
                values = [matrix[i, j] for i in class_indices for j in class_indices if i < j and np.isfinite(matrix[i, j])]
                rows.append(
                    {
                        "observation_id": observation_id,
                        "score_key": f"network_within::{anatomical_class}",
                        "score_label": f"{anatomical_class} within-network mean",
                        "score_family": "within_network_connectivity",
                        "delta": _safe_mean(values),
                    }
                )

        for i, class_a in enumerate(class_groups):
            indices_a = [idx for idx, roi in enumerate(roi_order) if classes.get(roi) == class_a]
            for class_b in class_groups[i + 1 :]:
                indices_b = [idx for idx, roi in enumerate(roi_order) if classes.get(roi) == class_b]
                values = [matrix[a, b] for a in indices_a for b in indices_b if np.isfinite(matrix[a, b])]
                rows.append(
                    {
                        "observation_id": observation_id,
                        "score_key": f"network_between::{class_a}<->{class_b}",
                        "score_label": f"{class_a} <-> {class_b}",
                        "score_family": "between_network_connectivity",
                        "delta": _safe_mean(values),
                    }
                )

        left_indices = [idx for idx, roi in enumerate(roi_order) if hemispheres.get(roi) == "left"]
        right_indices = [idx for idx, roi in enumerate(roi_order) if hemispheres.get(roi) == "right"]
        if left_indices and right_indices:
            left_strength = np.nanmean(np.nansum(matrix[left_indices, :], axis=1))
            right_strength = np.nanmean(np.nansum(matrix[right_indices, :], axis=1))
            li = np.nan
            denom = abs(left_strength) + abs(right_strength)
            if denom > 0:
                li = float((left_strength - right_strength) / denom)
            rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": "laterality::global",
                    "score_label": "Global laterality index",
                    "score_family": "laterality_index",
                    "delta": li,
                }
            )

        seen_pairs: set[tuple[str, str]] = set()
        for roi in roi_order:
            partner = homologues.get(roi)
            if not partner or (partner, roi) in seen_pairs:
                continue
            seen_pairs.add((roi, partner))
            i = roi_order.index(roi)
            j = roi_order.index(partner)
            roi_strength = np.nansum(matrix[i, :])
            partner_strength = np.nansum(matrix[j, :])
            denom = abs(roi_strength) + abs(partner_strength)
            li = np.nan if denom <= 0 else float((roi_strength - partner_strength) / denom)
            rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": f"laterality_pair::{roi}<->{partner}",
                    "score_label": f"Laterality {roi} vs {partner}",
                    "score_family": "laterality_pair_index",
                    "delta": li,
                }
            )

    return pd.DataFrame(rows)


def _analyze_network_scores(score_frame: pd.DataFrame, analysis_result: dict[str, Any]) -> dict[str, Any]:
    if score_frame.empty:
        return {"results": [], "message": "No network scores are available for the current selection."}

    frame = score_frame.copy()
    required = {"observation_id", "group_label", "delta"}
    if not required.issubset(frame.columns):
        return {"results": [], "message": "Observation metadata are not available for network analysis."}

    paired_longitudinal = (
        analysis_result.get("longitudinal_enabled", False)
        and frame.get("analysis_variant", pd.Series(dtype=object)).astype(str).eq("paired_longitudinal").any()
    )
    unpaired_longitudinal = (
        analysis_result.get("longitudinal_enabled", False)
        and not paired_longitudinal
        and "timepoint_label" in frame.columns
    )
    results: list[dict[str, Any]] = []

    for score_key, group in frame.groupby("score_key", sort=True):
        score_label = str(group["score_label"].iloc[0])
        score_family = str(group["score_family"].iloc[0])
        if unpaired_longitudinal:
            result = _summarize_unpaired_longitudinal_score(group, analysis_result)
        else:
            result = _summarize_standard_score(group, analysis_result)
        result["score_key"] = score_key
        result["score_label"] = score_label
        result["score_family"] = score_family
        results.append(result)

    results_frame = _attach_q_values(pd.DataFrame(results), analysis_result["correction_method"]) if results else pd.DataFrame()
    return {
        "results": results_frame.sort_values(["score_family", "score_label"]).to_dict(orient="records") if not results_frame.empty else [],
        "message": None if results else "No network score results are available.",
    }


def _summarize_standard_score(group: pd.DataFrame, analysis_result: dict[str, Any]) -> dict[str, Any]:
    between_groups = bool(analysis_result.get("between_groups"))
    group_a = str(analysis_result["selected_group_a"])
    group_b = str(analysis_result["selected_group_b"])
    if between_groups:
        first = group[group["group_label"].astype(str) == group_a]
        second = group[group["group_label"].astype(str) == group_b]
        delta_a = first["delta"].to_numpy(dtype=float)
        delta_b = second["delta"].to_numpy(dtype=float)
        n_a = len(delta_a)
        n_b = len(delta_b)
        if n_a >= 2 and n_b >= 2:
            statistic, p_value = stats.ttest_ind(delta_b, delta_a, equal_var=False)
            pooled_var = (((n_a - 1) * np.var(delta_a, ddof=1)) + ((n_b - 1) * np.var(delta_b, ddof=1))) / max(n_a + n_b - 2, 1)
            pooled_sd = float(np.sqrt(pooled_var)) if pooled_var >= 0 else np.nan
            effect_size = float((np.mean(delta_b) - np.mean(delta_a)) / pooled_sd) if pooled_sd and pooled_sd > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan
        mean_delta = float(np.mean(delta_b) - np.mean(delta_a)) if n_a and n_b else np.nan
        group_stats = [_group_summary(group_a, delta_a), _group_summary(group_b, delta_b)]
    else:
        deltas = group["delta"].to_numpy(dtype=float)
        n_obs = len(deltas)
        if n_obs >= 2:
            statistic, p_value = stats.ttest_1samp(deltas, popmean=0.0)
            std_delta = np.std(deltas, ddof=1)
            effect_size = float(np.mean(deltas) / std_delta) if std_delta > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan
        mean_delta = float(np.mean(deltas)) if n_obs else np.nan
        group_stats = [_group_summary(group_a, deltas)]
        n_a = n_obs
        n_b = n_obs

    return {
        "mean_delta": mean_delta,
        "effect_size": effect_size,
        "statistic": float(statistic) if pd.notna(statistic) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        "n": int(len(group)),
        "n_group_a": int(n_a),
        "n_group_b": int(n_b),
        "group_stats": group_stats,
        "detail_records": group.drop(columns=["score_key", "score_label", "score_family"], errors="ignore").to_dict(orient="records"),
    }


def _summarize_unpaired_longitudinal_score(group: pd.DataFrame, analysis_result: dict[str, Any]) -> dict[str, Any]:
    time_a = str(analysis_result["longitudinal_value_a"])
    time_b = str(analysis_result["longitudinal_value_b"])
    between_groups = bool(analysis_result.get("between_groups"))
    group_a = str(analysis_result["selected_group_a"])
    group_b = str(analysis_result["selected_group_b"])

    frame = group.copy()
    frame["timepoint_label"] = frame["timepoint_label"].astype(str)
    if between_groups:
        frame["group_indicator"] = (frame["group_label"].astype(str) == group_b).astype(int)
        frame["timepoint_indicator"] = (frame["timepoint_label"] == time_b).astype(int)
        frame["interaction"] = frame["group_indicator"] * frame["timepoint_indicator"]
        valid = frame.dropna(subset=["delta", "group_indicator", "timepoint_indicator", "interaction"])
        if len(valid) >= 4 and valid["interaction"].nunique() > 1 and valid["group_indicator"].nunique() > 1 and valid["timepoint_indicator"].nunique() > 1:
            predictors = sm.add_constant(valid[["group_indicator", "timepoint_indicator", "interaction"]], has_constant="add")
            model = sm.OLS(valid["delta"], predictors).fit()
            statistic = float(model.tvalues["interaction"])
            p_value = float(model.pvalues["interaction"])
            residual_sd = float(np.sqrt(model.mse_resid)) if model.df_resid > 0 else np.nan
            cell_means = valid.groupby(["group_label", "timepoint_label"])["delta"].mean()
            mean_delta = (
                _safe_value(cell_means.get((group_b, time_b)))
                - _safe_value(cell_means.get((group_b, time_a)))
                - _safe_value(cell_means.get((group_a, time_b)))
                + _safe_value(cell_means.get((group_a, time_a)))
            )
            effect_size = float(mean_delta / residual_sd) if residual_sd and residual_sd > 0 and pd.notna(mean_delta) else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            mean_delta = np.nan
            effect_size = np.nan
        group_stats = []
        for current_group, current_time in [(group_a, time_a), (group_a, time_b), (group_b, time_a), (group_b, time_b)]:
            values = frame[(frame["group_label"].astype(str) == current_group) & (frame["timepoint_label"] == current_time)]["delta"].to_numpy(dtype=float)
            group_stats.append(_group_summary(f"{current_group} | {current_time}", values))
        n_a = int(len(frame[frame["group_label"].astype(str) == group_a]))
        n_b = int(len(frame[frame["group_label"].astype(str) == group_b]))
    else:
        first = frame[frame["timepoint_label"] == time_a]["delta"].to_numpy(dtype=float)
        second = frame[frame["timepoint_label"] == time_b]["delta"].to_numpy(dtype=float)
        n_a = len(first)
        n_b = len(second)
        if n_a >= 2 and n_b >= 2:
            statistic, p_value = stats.ttest_ind(second, first, equal_var=False)
            pooled_var = (((n_a - 1) * np.var(first, ddof=1)) + ((n_b - 1) * np.var(second, ddof=1))) / max(n_a + n_b - 2, 1)
            pooled_sd = float(np.sqrt(pooled_var)) if pooled_var >= 0 else np.nan
            effect_size = float((np.mean(second) - np.mean(first)) / pooled_sd) if pooled_sd and pooled_sd > 0 else np.nan
        else:
            statistic = np.nan
            p_value = np.nan
            effect_size = np.nan
        mean_delta = float(np.mean(second) - np.mean(first)) if n_a and n_b else np.nan
        group_stats = [_group_summary(time_a, first), _group_summary(time_b, second)]

    return {
        "mean_delta": float(mean_delta) if pd.notna(mean_delta) else np.nan,
        "effect_size": effect_size,
        "statistic": float(statistic) if pd.notna(statistic) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        "n": int(len(frame)),
        "n_group_a": int(n_a),
        "n_group_b": int(n_b),
        "group_stats": group_stats,
        "detail_records": frame.drop(columns=["score_key", "score_label", "score_family"], errors="ignore").to_dict(orient="records"),
    }


def _compute_graph_metrics_for_observations(
    *,
    observation_frame: pd.DataFrame,
    observation_matrices: dict[str, np.ndarray],
    roi_order: list[str],
    roi_metadata: pd.DataFrame,
    config: NetworkAnalysisConfig,
) -> dict[str, pd.DataFrame]:
    global_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    for _, observation in observation_frame.iterrows():
        observation_id = str(observation["observation_id"])
        matrix = observation_matrices.get(observation_id)
        if matrix is None:
            continue
        weighted = _prepare_weight_matrix(matrix, config.weight_mode)
        thresholded = _apply_threshold(weighted, config.threshold_mode, config.threshold_value)
        node_metrics, global_metrics = _compute_single_graph_metrics(thresholded, roi_order)
        communities = _detect_components(thresholded)
        participation = _participation_coefficients(thresholded, communities, roi_order)
        modularity = _weighted_modularity(thresholded, communities)
        for roi, metrics in node_metrics.items():
            metrics["participation"] = float(participation.get(roi, np.nan))
        hubness = _hubness_scores(node_metrics)
        for roi, metrics in node_metrics.items():
            metrics["hubness"] = float(hubness.get(roi, np.nan))
            for metric_name, value in metrics.items():
                node_rows.append(
                    {
                        "observation_id": observation_id,
                        "score_key": f"{metric_name}::{roi}",
                        "score_label": f"{roi} {metric_name}",
                        "score_family": metric_name,
                        "delta": value,
                    }
                )
        for metric_name, value in global_metrics.items():
            global_rows.append(
                {
                    "observation_id": observation_id,
                    "score_key": f"{metric_name}::global",
                    "score_label": f"Global {metric_name}",
                    "score_family": metric_name,
                    "delta": value,
                }
            )
        global_rows.append(
            {
                "observation_id": observation_id,
                "score_key": "modularity::global",
                "score_label": "Global modularity",
                "score_family": "modularity",
                "delta": modularity,
            }
        )
    global_frame = pd.DataFrame(global_rows)
    node_frame = pd.DataFrame(node_rows)
    if not observation_frame.empty:
        metadata = observation_frame.copy()
        analysis_metadata_columns = [column for column in metadata.columns if column != "observation_id"]
        if not global_frame.empty:
            global_frame = global_frame.merge(metadata, on="observation_id", how="left", validate="many_to_one")
        if not node_frame.empty:
            node_frame = node_frame.merge(metadata, on="observation_id", how="left", validate="many_to_one")
        _ = analysis_metadata_columns
    return {"global_records": global_frame, "node_records": node_frame}


def _compute_single_graph_metrics(matrix: np.ndarray, roi_order: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    n = len(roi_order)
    node_metrics: dict[str, dict[str, float]] = {}
    degree = (matrix > 0).sum(axis=1).astype(float)
    strength = np.nansum(matrix, axis=1).astype(float)
    distances = _distance_matrix(matrix)
    for index, roi in enumerate(roi_order):
        node_metrics[roi] = {
            "node_strength": float(strength[index]),
            "degree": float(degree[index]),
            "clustering_coefficient": float(_weighted_clustering_for_node(matrix, index)),
            "local_efficiency": float(_local_efficiency(matrix, index)),
            "betweenness": float(_betweenness_centrality(matrix, index)),
        }
    global_eff = float(_global_efficiency_from_distances(distances))
    mean_local = float(np.nanmean([metrics["local_efficiency"] for metrics in node_metrics.values()])) if node_metrics else np.nan
    return node_metrics, {
        "global_efficiency": global_eff,
        "mean_local_efficiency": mean_local,
        "mean_strength": float(np.nanmean(strength)) if n else np.nan,
    }


def _compute_community_results(
    *,
    observation_frame: pd.DataFrame,
    observation_matrices: dict[str, np.ndarray],
    roi_order: list[str],
    roi_metadata: pd.DataFrame,
    config: NetworkAnalysisConfig,
    analysis_result: dict[str, Any],
) -> dict[str, Any]:
    if observation_frame.empty:
        return {"message": "No community analysis is available."}

    mean_matrices = _group_mean_matrices(observation_frame, observation_matrices, roi_order)
    communities: dict[str, Any] = {}
    for label, matrix in mean_matrices.items():
        prepared = _apply_threshold(_prepare_weight_matrix(matrix, config.weight_mode), config.threshold_mode, config.threshold_value)
        component_labels = _detect_components(prepared)
        communities[label] = {
            "matrix": prepared.tolist(),
            "components": [{"roi": roi, "component": int(component_labels[index])} for index, roi in enumerate(roi_order)],
            "component_count": int(len(set(component_labels.tolist()))),
        }

    return {
        "message": None,
        "note": "Community output is exploratory and based on connected suprathreshold components, not optimized modularity partitions.",
        "communities": communities,
        "comparison_labels": list(communities.keys()),
        "analysis_label": analysis_result["analysis_label"],
    }


def _compute_nbs_results(
    *,
    observation_frame: pd.DataFrame,
    observation_matrices: dict[str, np.ndarray],
    roi_order: list[str],
    config: NetworkAnalysisConfig,
    analysis_result: dict[str, Any],
) -> dict[str, Any]:
    if observation_frame.empty:
        return {"message": "No NBS observations are available."}

    edge_frame = _edge_frame_from_observations(observation_frame, observation_matrices, roi_order)
    if edge_frame.empty:
        return {"message": "No edge-level observations are available for NBS."}

    observed_stats = _nbs_edge_statistics(edge_frame, analysis_result)
    threshold = float(config.nbs_primary_threshold)
    suprathreshold = observed_stats[observed_stats["statistic"].abs() >= threshold].copy()
    components = _edge_components_from_stats(suprathreshold, roi_order)

    max_component_sizes: list[int] = []
    permutations = max(int(config.nbs_permutations), 10)
    for iteration in range(permutations):
        permuted = _permute_nbs_labels(edge_frame, analysis_result, seed=iteration)
        permuted_stats = _nbs_edge_statistics(permuted, analysis_result)
        perm_components = _edge_components_from_stats(permuted_stats[permuted_stats["statistic"].abs() >= threshold], roi_order)
        max_component_sizes.append(max((component["edge_count"] for component in perm_components), default=0))

    component_records = []
    for component in components:
        edge_count = int(component["edge_count"])
        p_value = float(np.mean([size >= edge_count for size in max_component_sizes])) if max_component_sizes else np.nan
        component_records.append({**component, "p_value": p_value})

    return {
        "message": None,
        "primary_threshold": threshold,
        "permutations": permutations,
        "components": component_records,
        "suprathreshold_edges": suprathreshold.to_dict(orient="records"),
    }


def _prepare_weight_matrix(matrix: np.ndarray, weight_mode: str) -> np.ndarray:
    prepared = np.nan_to_num(matrix.astype(float), nan=0.0)
    if weight_mode == "absolute":
        prepared = np.abs(prepared)
    elif weight_mode == "raw":
        prepared = prepared.copy()
        prepared[prepared < 0] = 0.0
    else:
        prepared = prepared.copy()
        prepared[prepared < 0] = 0.0
    np.fill_diagonal(prepared, 0.0)
    return prepared


def _apply_threshold(matrix: np.ndarray, threshold_mode: str, threshold_value: float) -> np.ndarray:
    thresholded = matrix.copy()
    if threshold_mode == "absolute weight":
        thresholded[thresholded < float(threshold_value)] = 0.0
    elif threshold_mode == "density":
        upper = thresholded[np.triu_indices_from(thresholded, k=1)]
        positive = upper[upper > 0]
        if len(positive):
            density = min(max(float(threshold_value), 0.0), 1.0)
            keep = max(int(np.ceil(len(positive) * density)), 1)
            cutoff = np.sort(positive)[-keep]
            thresholded[thresholded < cutoff] = 0.0
    np.fill_diagonal(thresholded, 0.0)
    return thresholded


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    graph = np.where(matrix > 0, 1.0 / np.maximum(matrix, 1e-12), 0.0)
    distances = shortest_path(graph, directed=False, unweighted=False)
    return np.asarray(distances, dtype=float)


def _weighted_clustering_for_node(matrix: np.ndarray, node: int) -> float:
    neighbors = np.where(matrix[node] > 0)[0]
    k = len(neighbors)
    if k < 2:
        return np.nan
    total = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            u = neighbors[i]
            v = neighbors[j]
            if matrix[u, v] <= 0:
                continue
            total += (matrix[node, u] * matrix[node, v] * matrix[u, v]) ** (1.0 / 3.0)
    return float((2.0 * total) / (k * (k - 1)))


def _global_efficiency_from_distances(distances: np.ndarray) -> float:
    if distances.size == 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(np.isfinite(distances) & (distances > 0), 1.0 / distances, 0.0)
    np.fill_diagonal(inverse, 0.0)
    n = len(distances)
    return float(inverse.sum() / max(n * (n - 1), 1))


def _local_efficiency(matrix: np.ndarray, node: int) -> float:
    neighbors = np.where(matrix[node] > 0)[0]
    if len(neighbors) < 2:
        return np.nan
    subgraph = matrix[np.ix_(neighbors, neighbors)]
    distances = _distance_matrix(subgraph)
    return _global_efficiency_from_distances(distances)


def _betweenness_centrality(matrix: np.ndarray, target: int) -> float:
    n = matrix.shape[0]
    if n < 3:
        return 0.0
    distances = _distance_matrix(matrix)
    count = 0.0
    total = 0.0
    for source in range(n):
        if source == target:
            continue
        for sink in range(source + 1, n):
            if sink == target:
                continue
            d_st = distances[source, sink]
            if not np.isfinite(d_st) or d_st <= 0:
                continue
            total += 1.0
            if abs((distances[source, target] + distances[target, sink]) - d_st) < 1e-8:
                count += 1.0
    return float(count / total) if total else 0.0


def _detect_components(matrix: np.ndarray) -> np.ndarray:
    adjacency = (matrix > 0).astype(int)
    component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
    _ = component_count
    return labels.astype(int)


def _participation_coefficients(matrix: np.ndarray, communities: np.ndarray, roi_order: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    strengths = np.nansum(matrix, axis=1)
    for node in range(matrix.shape[0]):
        strength = strengths[node]
        if strength <= 0:
            result[roi_order[node]] = np.nan
            continue
        coefficient = 1.0
        for community in np.unique(communities):
            mask = communities == community
            within_strength = np.nansum(matrix[node, mask])
            coefficient -= (within_strength / strength) ** 2
        result[roi_order[node]] = float(coefficient)
    return result


def _weighted_modularity(matrix: np.ndarray, communities: np.ndarray) -> float:
    total_weight = np.nansum(matrix) / 2.0
    if total_weight <= 0:
        return np.nan
    strengths = np.nansum(matrix, axis=1)
    modularity = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if communities[i] != communities[j]:
                continue
            modularity += matrix[i, j] - (strengths[i] * strengths[j]) / (2.0 * total_weight)
    return float(modularity / (2.0 * total_weight))


def _hubness_scores(node_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    if not node_metrics:
        return {}
    metric_names = ["node_strength", "degree", "betweenness", "participation"]
    ranks: dict[str, list[float]] = {roi: [] for roi in node_metrics}
    for metric_name in metric_names:
        values = pd.Series({roi: metrics.get(metric_name, np.nan) for roi, metrics in node_metrics.items()}, dtype=float)
        ranked = values.rank(method="average", pct=True)
        for roi, value in ranked.items():
            if pd.notna(value):
                ranks[str(roi)].append(float(value))
    return {roi: float(np.mean(values)) if values else np.nan for roi, values in ranks.items()}


def _group_mean_matrices(
    observation_frame: pd.DataFrame,
    observation_matrices: dict[str, np.ndarray],
    roi_order: list[str],
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    if "timepoint_label" in observation_frame.columns:
        observation_frame = observation_frame.copy()
        observation_frame["timepoint_label"] = observation_frame["timepoint_label"].astype(str)
    for label, group in observation_frame.groupby("group_label", sort=True):
        members = [observation_matrices[str(obs_id)] for obs_id in group["observation_id"] if str(obs_id) in observation_matrices]
        if members:
            stack = np.stack(members, axis=0)
            matrices[str(label)] = np.nanmean(stack, axis=0)
        else:
            matrices[str(label)] = np.zeros((len(roi_order), len(roi_order)))
    return matrices


def _edge_frame_from_observations(
    observation_frame: pd.DataFrame,
    observation_matrices: dict[str, np.ndarray],
    roi_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, observation in observation_frame.iterrows():
        observation_id = str(observation["observation_id"])
        matrix = observation_matrices.get(observation_id)
        if matrix is None:
            continue
        for i, roi_from in enumerate(roi_order):
            for j in range(i + 1, len(roi_order)):
                roi_to = roi_order[j]
                rows.append(
                    {
                        "observation_id": observation_id,
                        "roi_from": roi_from,
                        "roi_to": roi_to,
                        "delta": float(matrix[i, j]),
                        "group_label": observation.get("group_label"),
                        "timepoint_label": observation.get("timepoint_label"),
                    }
                )
    return pd.DataFrame(rows)


def _nbs_edge_statistics(edge_frame: pd.DataFrame, analysis_result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    between_groups = bool(analysis_result.get("between_groups"))
    unpaired_longitudinal = bool(analysis_result.get("longitudinal_enabled")) and "timepoint_label" in edge_frame.columns and edge_frame["timepoint_label"].notna().any()
    for (roi_from, roi_to), group in edge_frame.groupby(["roi_from", "roi_to"], sort=True):
        statistic = np.nan
        if unpaired_longitudinal:
            time_a = str(analysis_result["longitudinal_value_a"])
            time_b = str(analysis_result["longitudinal_value_b"])
            if between_groups:
                frame = group.copy()
                frame["group_indicator"] = (frame["group_label"].astype(str) == str(analysis_result["selected_group_b"])).astype(int)
                frame["timepoint_indicator"] = (frame["timepoint_label"].astype(str) == time_b).astype(int)
                frame["interaction"] = frame["group_indicator"] * frame["timepoint_indicator"]
                if len(frame) >= 4 and frame["interaction"].nunique() > 1 and frame["group_indicator"].nunique() > 1 and frame["timepoint_indicator"].nunique() > 1:
                    predictors = sm.add_constant(frame[["group_indicator", "timepoint_indicator", "interaction"]], has_constant="add")
                    statistic = float(sm.OLS(frame["delta"], predictors).fit().tvalues["interaction"])
            else:
                first = group[group["timepoint_label"].astype(str) == time_a]["delta"].to_numpy(dtype=float)
                second = group[group["timepoint_label"].astype(str) == time_b]["delta"].to_numpy(dtype=float)
                if len(first) >= 2 and len(second) >= 2:
                    statistic = float(stats.ttest_ind(second, first, equal_var=False).statistic)
        else:
            if between_groups:
                first = group[group["group_label"].astype(str) == str(analysis_result["selected_group_a"])]["delta"].to_numpy(dtype=float)
                second = group[group["group_label"].astype(str) == str(analysis_result["selected_group_b"])]["delta"].to_numpy(dtype=float)
                if len(first) >= 2 and len(second) >= 2:
                    statistic = float(stats.ttest_ind(second, first, equal_var=False).statistic)
            else:
                values = group["delta"].to_numpy(dtype=float)
                if len(values) >= 2:
                    statistic = float(stats.ttest_1samp(values, popmean=0.0).statistic)
        rows.append({"roi_from": roi_from, "roi_to": roi_to, "statistic": statistic})
    return pd.DataFrame(rows)


def _edge_components_from_stats(stats_frame: pd.DataFrame, roi_order: list[str]) -> list[dict[str, Any]]:
    if stats_frame.empty:
        return []
    adjacency = np.zeros((len(roi_order), len(roi_order)), dtype=int)
    lookup = {roi: index for index, roi in enumerate(roi_order)}
    for _, row in stats_frame.iterrows():
        i = lookup[str(row["roi_from"])]
        j = lookup[str(row["roi_to"])]
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    components: list[dict[str, Any]] = []
    for component in sorted(set(labels.tolist())):
        nodes = [roi_order[index] for index, label in enumerate(labels.tolist()) if label == component]
        edges = [
            {"roi_from": row["roi_from"], "roi_to": row["roi_to"], "statistic": float(row["statistic"])}
            for _, row in stats_frame.iterrows()
            if row["roi_from"] in nodes and row["roi_to"] in nodes
        ]
        if not edges:
            continue
        components.append({"nodes": nodes, "edges": edges, "edge_count": len(edges)})
    return components


def _permute_nbs_labels(edge_frame: pd.DataFrame, analysis_result: dict[str, Any], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    permuted = edge_frame.copy()
    between_groups = bool(analysis_result.get("between_groups"))
    unpaired_longitudinal = bool(analysis_result.get("longitudinal_enabled")) and "timepoint_label" in permuted.columns and permuted["timepoint_label"].notna().any()
    if not between_groups and not unpaired_longitudinal:
        signs = pd.Series(rng.choice([-1.0, 1.0], size=permuted["observation_id"].nunique()), index=sorted(permuted["observation_id"].unique()))
        permuted["delta"] = permuted["delta"] * permuted["observation_id"].map(signs)
        return permuted
    if between_groups and not unpaired_longitudinal:
        shuffled = rng.permutation(permuted[["observation_id", "group_label"]].drop_duplicates()["group_label"].to_numpy())
        mapping = dict(zip(sorted(permuted["observation_id"].unique()), shuffled))
        permuted["group_label"] = permuted["observation_id"].map(mapping)
        return permuted
    labels = permuted[["observation_id", "group_label", "timepoint_label"]].drop_duplicates().reset_index(drop=True)
    shuffled = labels.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled.index = labels["observation_id"]
    permuted["group_label"] = permuted["observation_id"].map(shuffled["group_label"].to_dict())
    permuted["timepoint_label"] = permuted["observation_id"].map(shuffled["timepoint_label"].to_dict())
    return permuted


def _safe_mean(values: Any) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _safe_value(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)
