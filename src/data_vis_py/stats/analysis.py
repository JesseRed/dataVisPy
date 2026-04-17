"""Connectivity analysis helpers used by the Dash UI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
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


def run_analysis(bundle: DatasetBundle, config: AnalysisConfig) -> dict[str, Any]:
    """Run an analysis and return matrix and detail records."""
    filtered = bundle.connectivity[
        (bundle.connectivity["dataset_id"] == config.dataset_id)
        & (bundle.connectivity["metric"] == config.metric)
        & (bundle.connectivity["freq"] >= config.freq_min)
        & (bundle.connectivity["freq"] <= config.freq_max)
    ].copy()
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
    diagnostics: dict[str, Any] = {
        "observed": regression_frame[response_column].astype(float).tolist(),
        "fitted": model.fittedvalues.astype(float).tolist(),
        "residuals": model.resid.astype(float).tolist(),
        "point_labels": point_labels,
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
    sem = float(stats.sem(deltas, nan_policy="omit")) if n_obs >= 2 else np.nan
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
