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


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for a single analysis request."""

    analysis_mode: str
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
        test_label = "Paired t-test on within-subject trial deltas" if not group_context["between_groups"] else "Welch t-test between groups on longitudinal deltas"
    elif config.analysis_mode == "trial_delta":
        if config.mtime_filter != "All":
            filtered = filtered[filtered["mtime"] == config.mtime_filter]
        results, group_context = _trial_delta_results(filtered, config)
        analysis_label = f"Trial {config.trial_b} - Trial {config.trial_a}"
        test_label = "Paired t-test" if not group_context["between_groups"] else "Welch t-test between groups on trial deltas"
    else:
        results, group_context = _session_delta_results(filtered, config)
        analysis_label = f"M2 - M1 at Trial {config.trial_a}"
        test_label = "Paired t-test" if not group_context["between_groups"] else "Welch t-test between groups on session deltas"

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
        merged = detail_frame.merge(bundle.subjects, on="subject_id", how="left")
    else:
        baseline = (
            bundle.subjects.assign(mtime_order=bundle.subjects["mtime"].str.extract(r"(\d+)").fillna("999").astype(int))
            .sort_values(["base_subject_id", "mtime_order"])
            .drop_duplicates("base_subject_id", keep="first")
            .drop(columns="mtime_order")
        )
        join_columns = ["base_subject_id"]
        merged = detail_frame.merge(baseline, on="base_subject_id", how="left")

    result: dict[str, Any] = {"joined_columns": join_columns, "message": None}

    if correlation_variable:
        corr_frame = merged[["delta", correlation_variable]].dropna()
        if len(corr_frame) >= 3:
            if correlation_method == "spearman":
                corr = stats.spearmanr(corr_frame["delta"], corr_frame[correlation_variable])
                corr_value = float(corr.statistic)
                p_value = float(corr.pvalue)
            else:
                corr_value, p_value = stats.pearsonr(corr_frame["delta"], corr_frame[correlation_variable])
            result["correlation"] = {
                "variable": correlation_variable,
                "method": correlation_method,
                "n": int(len(corr_frame)),
                "statistic": float(corr_value),
                "p_value": float(p_value),
                "x": corr_frame[correlation_variable].tolist(),
                "y": corr_frame["delta"].tolist(),
            }
        else:
            result["correlation"] = {
                "variable": correlation_variable,
                "method": correlation_method,
                "n": int(len(corr_frame)),
                "message": "Not enough complete observations for correlation.",
            }

    covariates = regression_covariates or []
    if covariates:
        regression_frame = merged[["delta"] + covariates].dropna()
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


def _session_delta_results(filtered: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    comparison = filtered[filtered["trial_id"] == config.trial_a].copy()
    pivoted = (
        comparison.pivot_table(
            index=["base_subject_id", "group_label", "roi_from", "roi_to"],
            columns="mtime",
            values="value",
        )
        .dropna(subset=["M1", "M2"])
        .reset_index()
    )
    pivoted["delta"] = pivoted["M2"] - pivoted["M1"]

    detail_lookup = (
        comparison[["subject_id", "base_subject_id", "roi_from", "roi_to", "mtime", "value"]]
        .drop_duplicates()
    )

    detail_records = []
    for _, row in pivoted.iterrows():
        pair_subjects = detail_lookup[
            (detail_lookup["base_subject_id"] == row["base_subject_id"])
            & (detail_lookup["roi_from"] == row["roi_from"])
            & (detail_lookup["roi_to"] == row["roi_to"])
        ]
        m1_record = pair_subjects[pair_subjects["mtime"] == "M1"].head(1)
        m2_record = pair_subjects[pair_subjects["mtime"] == "M2"].head(1)
        detail_records.append(
            {
                "base_subject_id": row["base_subject_id"],
                "idx": row["base_subject_id"],
                "group_label": row["group_label"],
                "roi_from": row["roi_from"],
                "roi_to": row["roi_to"],
                "subject_id_m1": m1_record["subject_id"].iloc[0] if not m1_record.empty else None,
                "subject_id_m2": m2_record["subject_id"].iloc[0] if not m2_record.empty else None,
                "m1_value": float(row["M1"]),
                "m2_value": float(row["M2"]),
                "delta": float(row["delta"]),
            }
        )
    detail_frame = pd.DataFrame(detail_records)
    return _analyze_by_group_selection(
        detail_frame,
        config,
        detail_columns=["base_subject_id", "idx", "group_label", "subject_id_m1", "subject_id_m2", "m1_value", "m2_value", "delta"],
        mode="session_delta",
    )


def _longitudinal_trial_delta_results(filtered: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
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


def _analyze_by_group_selection(frame: pd.DataFrame, config: AnalysisConfig, detail_columns: list[str], mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_a = str(config.group_a) if config.group_a is not None else None
    group_b = str(config.group_b) if config.group_b is not None else None
    if group_a is None and not frame.empty:
        group_a = str(frame["group_label"].dropna().iloc[0])
    if group_b is None:
        group_b = group_a

    frame = frame.copy()
    frame["group_label"] = frame["group_label"].astype(str)

    if group_a == group_b:
        subset = frame[frame["group_label"] == group_a] if group_a is not None else frame
        results = _within_group_results(subset, detail_columns, mode)
        context = {"group_a": group_a, "group_b": group_b, "between_groups": False}
    else:
        subset = frame[frame["group_label"].isin([group_a, group_b])]
        results = _between_group_results(subset, detail_columns, group_a, group_b)
        context = {"group_a": group_a, "group_b": group_b, "between_groups": True}
    return _attach_q_values(results, config.correction_method), context


def _within_group_results(frame: pd.DataFrame, detail_columns: list[str], mode: str) -> pd.DataFrame:
    results = []
    for (roi_from, roi_to), group in frame.groupby(["roi_from", "roi_to"], sort=True):
        deltas = group["delta"].to_numpy(dtype=float)
        n_obs = len(group)
        group_name = str(group["group_label"].iloc[0]) if n_obs else "Group 1"
        if mode == "trial_delta":
            a_values = group["trial_a_value"].to_numpy(dtype=float)
            b_values = group["trial_b_value"].to_numpy(dtype=float)
        elif mode == "session_delta":
            a_values = group["m1_value"].to_numpy(dtype=float)
            b_values = group["m2_value"].to_numpy(dtype=float)
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
