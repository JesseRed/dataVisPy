"""Tests for the core connectivity analysis routines."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_vis_py.io.dataset_loader import DatasetBundle, load_dataset
from data_vis_py.stats.analysis import (
    ALL_GROUPS_LABEL,
    ALL_GROUPS_VALUE,
    AnalysisConfig,
    LongitudinalDetailConfig,
    NetworkAnalysisConfig,
    PatternsAnalysisConfig,
    SELECTED_EDGE_DELTA_VALUE,
    build_patterns_feature_data,
    derive_roi_metadata,
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
from data_vis_py.ui.dashboard import create_dashboard


def make_regression_test_bundle() -> DatasetBundle:
    subjects = pd.DataFrame(
        [
            {"dataset_id": "synthetic", "subject_id": "A1_M1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "age": 10.0, "score": 1.0, "bmi": 20.0, "weight": 60.0},
            {"dataset_id": "synthetic", "subject_id": "A1_M2", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M2", "age": 10.0, "score": 1.0, "bmi": 20.0, "weight": 60.0},
            {"dataset_id": "synthetic", "subject_id": "A2_M1", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "age": 20.0, "score": 2.0, "bmi": 21.0, "weight": 65.0},
            {"dataset_id": "synthetic", "subject_id": "A2_M2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M2", "age": 20.0, "score": 2.0, "bmi": 21.0, "weight": 65.0},
            {"dataset_id": "synthetic", "subject_id": "B1_M1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "age": 10.0, "score": 1.5, "bmi": 22.0, "weight": 70.0},
            {"dataset_id": "synthetic", "subject_id": "B1_M2", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M2", "age": 10.0, "score": 1.5, "bmi": 22.0, "weight": 70.0},
            {"dataset_id": "synthetic", "subject_id": "B2_M1", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "age": 20.0, "score": 2.5, "bmi": 23.0, "weight": 75.0},
            {"dataset_id": "synthetic", "subject_id": "B2_M2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M2", "age": 20.0, "score": 2.5, "bmi": 23.0, "weight": 75.0},
            {"dataset_id": "synthetic", "subject_id": "A3_M1", "idx": "5", "base_subject_id": "5", "group_label": "A", "mtime": "M1", "age": 30.0, "score": 3.0, "bmi": 24.0, "weight": 80.0},
            {"dataset_id": "synthetic", "subject_id": "A3_M2", "idx": "5", "base_subject_id": "5", "group_label": "A", "mtime": "M2", "age": 30.0, "score": 3.0, "bmi": 24.0, "weight": 80.0},
            {"dataset_id": "synthetic", "subject_id": "A4_M2", "idx": "6", "base_subject_id": "6", "group_label": "A", "mtime": "M2", "age": 40.0, "score": 4.0, "bmi": 25.0, "weight": 85.0},
            {"dataset_id": "synthetic", "subject_id": "B3_M1", "idx": "7", "base_subject_id": "7", "group_label": "B", "mtime": "M1", "age": 30.0, "score": 3.5, "bmi": 26.0, "weight": 90.0},
        ]
    )
    delta_lookup = {
        "A1_M1": 1.0,
        "A1_M2": 2.0,
        "A2_M1": 2.0,
        "A2_M2": 3.0,
        "B1_M1": 1.0,
        "B1_M2": 4.0,
        "B2_M1": 1.5,
        "B2_M2": 5.0,
        "A3_M1": 2.5,
        "A3_M2": 4.0,
        "A4_M2": 4.5,
        "B3_M1": 2.0,
    }
    connectivity_rows = []
    for subject_id, delta in delta_lookup.items():
        subject_row = subjects.loc[subjects["subject_id"] == subject_id].iloc[0]
        connectivity_rows.append(
            {
                "dataset_id": "synthetic",
                "metric": "conn_coh",
                "subject_id": subject_id,
                "idx": subject_row["idx"],
                "base_subject_id": subject_row["base_subject_id"],
                "group_label": subject_row["group_label"],
                "mtime": subject_row["mtime"],
                "trial_id": 1,
                "freq": 10.0,
                "roi_from": "roi_a",
                "roi_to": "roi_b",
                "value": 0.0,
            }
        )
        connectivity_rows.append(
            {
                "dataset_id": "synthetic",
                "metric": "conn_coh",
                "subject_id": subject_id,
                "idx": subject_row["idx"],
                "base_subject_id": subject_row["base_subject_id"],
                "group_label": subject_row["group_label"],
                "mtime": subject_row["mtime"],
                "trial_id": 2,
                "freq": 10.0,
                "roi_from": "roi_a",
                "roi_to": "roi_b",
                "value": float(delta),
            }
        )
    return DatasetBundle(
        dataset_id="synthetic",
        metric="conn_coh",
        channels=["roi_a", "roi_b"],
        trial_ids=[1, 2],
        frequencies=[10.0],
        subjects=subjects,
        connectivity=pd.DataFrame(connectivity_rows),
    )


def make_outlier_test_bundle() -> DatasetBundle:
    subjects = pd.DataFrame(
        [
            {"dataset_id": "synthetic", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "age": 10.0},
            {"dataset_id": "synthetic", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "age": 20.0},
            {"dataset_id": "synthetic", "subject_id": "B1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "age": 30.0},
            {"dataset_id": "synthetic", "subject_id": "B2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "age": 40.0},
        ]
    )
    deltas = {"A1": 1.0, "A2": 1.0, "B1": 1.0, "B2": 10.0}
    connectivity_rows = []
    for subject_id, delta in deltas.items():
        subject_row = subjects.loc[subjects["subject_id"] == subject_id].iloc[0]
        connectivity_rows.extend(
            [
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 1,
                    "freq": 10.0,
                    "roi_from": "roi_a",
                    "roi_to": "roi_b",
                    "value": 0.0,
                },
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 2,
                    "freq": 10.0,
                    "roi_from": "roi_a",
                    "roi_to": "roi_b",
                    "value": float(delta),
                },
            ]
        )
    return DatasetBundle(
        dataset_id="synthetic",
        metric="conn_coh",
        channels=["roi_a", "roi_b"],
        trial_ids=[1, 2],
        frequencies=[10.0],
        subjects=subjects,
        connectivity=pd.DataFrame(connectivity_rows),
    )


def make_longitudinal_detail_bundle() -> DatasetBundle:
    subjects = pd.DataFrame(
        [
            {"dataset_id": "synthetic", "subject_id": "A1_M1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "age": 50.0},
            {"dataset_id": "synthetic", "subject_id": "A1_M2", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M2", "age": 50.0},
            {"dataset_id": "synthetic", "subject_id": "A1_M3", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M3", "age": 50.0},
            {"dataset_id": "synthetic", "subject_id": "A2_M1", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "age": 55.0},
            {"dataset_id": "synthetic", "subject_id": "A2_M2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M2", "age": 55.0},
            {"dataset_id": "synthetic", "subject_id": "A2_M3", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M3", "age": 55.0},
            {"dataset_id": "synthetic", "subject_id": "B1_M1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "age": 60.0},
            {"dataset_id": "synthetic", "subject_id": "B1_M2", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M2", "age": 60.0},
            {"dataset_id": "synthetic", "subject_id": "B1_M3", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M3", "age": 60.0},
            {"dataset_id": "synthetic", "subject_id": "B2_M1", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "age": 65.0},
            {"dataset_id": "synthetic", "subject_id": "B2_M2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M2", "age": 65.0},
            {"dataset_id": "synthetic", "subject_id": "B2_M3", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M3", "age": 65.0},
        ]
    )
    delta_lookup = {
        "A1_M1": 1.0,
        "A1_M2": 1.5,
        "A1_M3": 2.0,
        "A2_M1": 1.2,
        "A2_M2": 1.8,
        "A2_M3": 2.3,
        "B1_M1": 1.0,
        "B1_M2": 2.4,
        "B1_M3": 3.2,
        "B2_M1": 1.1,
        "B2_M2": 2.6,
        "B2_M3": 3.4,
    }
    connectivity_rows = []
    for subject_id, delta in delta_lookup.items():
        subject_row = subjects.loc[subjects["subject_id"] == subject_id].iloc[0]
        connectivity_rows.extend(
            [
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 1,
                    "freq": 10.0,
                    "roi_from": "roi_a",
                    "roi_to": "roi_b",
                    "value": 0.0,
                },
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 2,
                    "freq": 10.0,
                    "roi_from": "roi_a",
                    "roi_to": "roi_b",
                    "value": float(delta),
                },
            ]
        )
    return DatasetBundle(
        dataset_id="synthetic",
        metric="conn_coh",
        channels=["roi_a", "roi_b"],
        trial_ids=[1, 2],
        frequencies=[10.0],
        subjects=subjects,
        connectivity=pd.DataFrame(connectivity_rows),
    )


def make_network_test_bundle() -> DatasetBundle:
    subjects = pd.DataFrame(
        [
            {"dataset_id": "synthetic", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "age": 55.0, "score": 3.0},
            {"dataset_id": "synthetic", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "age": 61.0, "score": 4.0},
            {"dataset_id": "synthetic", "subject_id": "B1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "age": 70.0, "score": 8.0},
            {"dataset_id": "synthetic", "subject_id": "B2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "age": 72.0, "score": 7.0},
        ]
    )
    rois = ["frontal_left", "frontal_right", "parietal_left", "parietal_right"]
    pair_values = {
        "A1": {
            ("frontal_left", "frontal_right"): 0.9,
            ("frontal_left", "parietal_left"): 0.8,
            ("frontal_left", "parietal_right"): 0.3,
            ("frontal_right", "parietal_left"): 0.2,
            ("frontal_right", "parietal_right"): 0.4,
            ("parietal_left", "parietal_right"): 0.7,
        },
        "A2": {
            ("frontal_left", "frontal_right"): 0.8,
            ("frontal_left", "parietal_left"): 0.75,
            ("frontal_left", "parietal_right"): 0.25,
            ("frontal_right", "parietal_left"): 0.15,
            ("frontal_right", "parietal_right"): 0.35,
            ("parietal_left", "parietal_right"): 0.65,
        },
        "B1": {
            ("frontal_left", "frontal_right"): 0.2,
            ("frontal_left", "parietal_left"): 0.1,
            ("frontal_left", "parietal_right"): 0.45,
            ("frontal_right", "parietal_left"): 0.55,
            ("frontal_right", "parietal_right"): 0.85,
            ("parietal_left", "parietal_right"): 0.5,
        },
        "B2": {
            ("frontal_left", "frontal_right"): 0.25,
            ("frontal_left", "parietal_left"): 0.15,
            ("frontal_left", "parietal_right"): 0.4,
            ("frontal_right", "parietal_left"): 0.5,
            ("frontal_right", "parietal_right"): 0.8,
            ("parietal_left", "parietal_right"): 0.45,
        },
    }
    connectivity_rows = []
    for subject_id, pairs in pair_values.items():
        subject_row = subjects.loc[subjects["subject_id"] == subject_id].iloc[0]
        for (roi_from, roi_to), value in pairs.items():
            connectivity_rows.append(
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 1,
                    "freq": 10.0,
                    "roi_from": roi_from,
                    "roi_to": roi_to,
                    "value": 0.0,
                }
            )
            connectivity_rows.append(
                {
                    "dataset_id": "synthetic",
                    "metric": "conn_coh",
                    "subject_id": subject_id,
                    "idx": subject_row["idx"],
                    "base_subject_id": subject_row["base_subject_id"],
                    "group_label": subject_row["group_label"],
                    "mtime": subject_row["mtime"],
                    "trial_id": 2,
                    "freq": 10.0,
                    "roi_from": roi_from,
                    "roi_to": roi_to,
                    "value": float(value),
                }
            )
    return DatasetBundle(
        dataset_id="synthetic",
        metric="conn_coh",
        channels=rois,
        trial_ids=[1, 2],
        frequencies=[10.0],
        subjects=subjects,
        connectivity=pd.DataFrame(connectivity_rows),
    )


class AnalysisTests(unittest.TestCase):
    """Checks for matrix analysis and covariate calculations."""

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = repo_root / "data" / "raw" / "REST_24_Stroke"
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.bundle = load_dataset(dataset_dir, cache_root=cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def test_trial_delta_analysis_returns_expected_shapes(self) -> None:
        config = AnalysisConfig(
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=self.bundle.frequencies[15],
            freq_max=self.bundle.frequencies[15],
            mtime_filter="All",
            correction_method="fdr_bh",
        )
        result = run_analysis(self.bundle, config)
        self.assertEqual(len(result["pair_results"]), 45)
        self.assertEqual(len(result["matrix"]), 10)
        self.assertEqual(len(result["matrix"][0]), 10)
        self.assertIn("freq_label", result)
        self.assertIn("test_label", result)
        self.assertIn("correction_label", result)
        first_pair = result["pair_results"][0]
        self.assertIn("q_value", first_pair)
        self.assertIn("q_value_fdr_bh", first_pair)
        self.assertIn("q_value_bonferroni", first_pair)
        self.assertIn("q_value_holm", first_pair)
        self.assertIn("group_stats", first_pair)
        self.assertIn("detail_records", first_pair)
        self.assertGreaterEqual(first_pair["n"], 1)

    def test_detail_summary_and_covariate_analysis_are_consistent(self) -> None:
        config = AnalysisConfig(
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=self.bundle.frequencies[20],
            freq_max=self.bundle.frequencies[20],
            mtime_filter="M1",
            correction_method="fdr_bh",
        )
        result = run_analysis(self.bundle, config)
        pair_key = result["pair_results"][0]["pair_key"]
        summary = summarize_pair_result(result, pair_key)
        self.assertIsNotNone(summary)
        self.assertEqual(len(summary["detail_records"]), summary["n"])

        covariate_result = run_covariate_analysis(
            self.bundle,
            result,
            pair_key,
            correlation_method="pearson",
            correlation_variable="age",
            regression_covariates=["age", "MoCA"],
        )
        self.assertIn("correlation", covariate_result)
        self.assertIn("regression", covariate_result)

    def test_frequency_band_average_matches_manual_mean(self) -> None:
        freq_min = self.bundle.frequencies[15]
        freq_max = self.bundle.frequencies[17]
        config = AnalysisConfig(
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=freq_min,
            freq_max=freq_max,
            mtime_filter="M1",
            correction_method="fdr_bh",
        )
        result = run_analysis(self.bundle, config)
        pair_key = result["pair_results"][0]["pair_key"]
        roi_from, roi_to = pair_key.split("|")
        detail_record = result["pair_results"][0]["detail_records"][0]
        subject_id = detail_record["subject_id"]

        manual = self.bundle.connectivity[
            (self.bundle.connectivity["subject_id"] == subject_id)
            & (self.bundle.connectivity["roi_from"] == roi_from)
            & (self.bundle.connectivity["roi_to"] == roi_to)
            & (self.bundle.connectivity["trial_id"].isin([self.bundle.trial_ids[0], self.bundle.trial_ids[1]]))
            & (self.bundle.connectivity["freq"] >= freq_min)
            & (self.bundle.connectivity["freq"] <= freq_max)
        ]
        manual_means = manual.groupby("trial_id")["value"].mean()
        self.assertAlmostEqual(detail_record["trial_a_value"], manual_means[self.bundle.trial_ids[0]])
        self.assertAlmostEqual(detail_record["trial_b_value"], manual_means[self.bundle.trial_ids[1]])

    def test_none_correction_keeps_q_values_equal_to_p_values(self) -> None:
        config = AnalysisConfig(
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=self.bundle.frequencies[18],
            freq_max=self.bundle.frequencies[22],
            mtime_filter="All",
            correction_method="none",
        )
        result = run_analysis(self.bundle, config)
        self.assertEqual(result["correction_label"], "None (raw p-values)")
        for pair in result["pair_results"]:
            if pair["p_value"] == pair["p_value"]:
                self.assertAlmostEqual(pair["p_value"], pair["q_value"])

    def test_excluded_idx_removes_all_measurements_for_that_person(self) -> None:
        bundle = make_regression_test_bundle()
        result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="All",
                correction_method="none",
                group_a=ALL_GROUPS_VALUE,
                group_b=ALL_GROUPS_VALUE,
                excluded_idx=("1",),
            ),
        )
        pair = result["pair_results"][0]
        observed_idx = {str(record["idx"]) for record in pair["detail_records"]}
        self.assertNotIn("1", observed_idx)
        self.assertEqual(pair["n"], 10)

    def test_leave_one_out_returns_one_run_per_remaining_idx_and_ranks_outlier(self) -> None:
        bundle = make_outlier_test_bundle()
        config = AnalysisConfig(
            dataset_id="synthetic",
            metric="conn_coh",
            trial_a=1,
            trial_b=2,
            freq_min=10.0,
            freq_max=10.0,
            mtime_filter="M1",
            correction_method="none",
            group_a="A",
            group_b="B",
        )
        result = run_leave_one_out_analysis(
            bundle,
            config,
            "roi_a|roi_b",
            significance_threshold=0.05,
            outcome_variable=SELECTED_EDGE_DELTA_VALUE,
            regression_covariates=["age"],
        )
        self.assertEqual(len(result["global_records"]), 4)
        self.assertEqual(len(result["pair_records"]), 4)
        self.assertEqual({row["excluded_idx"] for row in result["global_records"][:2]}, {"3", "4"})
        self.assertEqual({row["excluded_idx"] for row in result["pair_records"][:2]}, {"3", "4"})
        self.assertIn("4", result["top3_idx"])

    def test_regression_influence_contains_expected_diagnostic_lengths(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_regression_influence_analysis(
            bundle,
            analysis_result,
            "roi_a|roi_b",
            outcome_variable=SELECTED_EDGE_DELTA_VALUE,
            regression_covariates=["age"],
        )
        diagnostics = result["diagnostics"]
        self.assertEqual(len(diagnostics["observed"]), result["n"])
        self.assertEqual(len(diagnostics["leverage"]), result["n"])
        self.assertEqual(len(diagnostics["cooks_distance"]), result["n"])
        self.assertEqual(len(diagnostics["studentized_residuals"]), result["n"])
        self.assertEqual(len(diagnostics["observation_rows"]), result["n"])
        self.assertTrue(all("idx" in row for row in diagnostics["observation_rows"]))

    def test_derive_roi_metadata_infers_hemisphere_class_and_homologue(self) -> None:
        metadata = derive_roi_metadata(["frontal_left", "frontal_right", "occipital"])
        self.assertEqual(metadata[0]["hemisphere"], "left")
        self.assertEqual(metadata[0]["anatomical_class"], "frontal")
        self.assertEqual(metadata[0]["homologue"], "frontal_right")
        self.assertEqual(metadata[2]["hemisphere"], "midline")
        self.assertIsNone(metadata[2]["homologue"])

    def test_network_analysis_returns_summary_graph_module_and_nbs_outputs(self) -> None:
        bundle = make_network_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_network_analysis(
            bundle,
            analysis_result,
            NetworkAnalysisConfig(
                mode="summary",
                metric_name="roi_mean_connectivity",
                threshold_mode="absolute weight",
                threshold_value=0.3,
                nbs_permutations=20,
            ),
        )
        summary_records = result["network_summary_results"]["results"]
        self.assertTrue(any(record["score_family"] == "roi_mean_connectivity" for record in summary_records))
        self.assertTrue(any(record["score_family"] == "within_network_connectivity" for record in summary_records))
        self.assertTrue(any(record["score_family"] == "laterality_index" for record in summary_records))
        self.assertTrue(result["graph_metric_results"]["global"]["results"])
        self.assertTrue(result["graph_metric_results"]["node"]["results"])
        self.assertIn("communities", result["community_results"])
        self.assertIn("components", result["nbs_results"])

    def test_network_thresholding_changes_degree_metrics(self) -> None:
        bundle = make_network_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        baseline = run_network_analysis(bundle, analysis_result, NetworkAnalysisConfig(metric_name="degree", threshold_mode="none", nbs_permutations=20))
        thresholded = run_network_analysis(
            bundle,
            analysis_result,
            NetworkAnalysisConfig(metric_name="degree", threshold_mode="absolute weight", threshold_value=0.7, nbs_permutations=20),
        )
        baseline_degree = next(record for record in baseline["graph_metric_results"]["node"]["results"] if record["score_family"] == "degree")
        thresholded_degree = next(record for record in thresholded["graph_metric_results"]["node"]["results"] if record["score_family"] == "degree")
        self.assertNotEqual(baseline_degree["mean_delta"], thresholded_degree["mean_delta"])

    def test_network_summary_contains_global_laterality_score(self) -> None:
        bundle = make_network_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_network_analysis(bundle, analysis_result, NetworkAnalysisConfig(metric_name="laterality_index", nbs_permutations=20))
        labels = {record["score_key"] for record in result["network_summary_results"]["results"]}
        self.assertIn("laterality::global", labels)

    def test_dashboard_layout_contains_network_tab(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = repo_root / "data" / "raw" / "REST_24_Stroke"
        app = create_dashboard(
            dataset_dir=dataset_dir,
            initial_bundle=self.bundle,
            json_files=["data_coh.json"],
            csv_files=["info.csv"],
            initial_json="data_coh.json",
            initial_csv="info.csv",
        )
        main_tabs = app.layout.children[4].children[0]
        tab_labels = [tab.label for tab in main_tabs.children]
        self.assertIn("Network", tab_labels)
        self.assertIn("Patterns", tab_labels)
        self.assertNotIn("Covariates", tab_labels)
        layout_repr = repr(app.layout)
        self.assertIn("heatmap-longitudinal-model", layout_repr)
        self.assertIn("heatmap-longitudinal-fit-chart", layout_repr)
        self.assertIn("heatmap-reliable-change-chart", layout_repr)

    def test_longitudinal_detail_analysis_returns_models_trajectory_and_reliable_change(self) -> None:
        bundle = make_longitudinal_detail_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
                longitudinal_enabled=True,
                longitudinal_require_pairs=False,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_longitudinal_detail_analysis(
            bundle,
            analysis_result,
            "roi_a|roi_b",
            config=LongitudinalDetailConfig(
                model_family="mixed_effects",
                random_slope_time=True,
                baseline_value="M1",
                followup_value="M2",
            ),
            regression_covariates=["age"],
        )
        self.assertIsNone(result["message"])
        self.assertIn("primary_model", result)
        self.assertIn("change_score", result)
        self.assertIn("ancova", result)
        self.assertIn("trajectory", result)
        self.assertIn("reliable_change", result)
        self.assertFalse(result["primary_model"].get("random_slope_used", False))
        self.assertEqual(result["trajectory"]["ordered_timepoints"], ["M1", "M2", "M3"])
        self.assertTrue(result["reliable_change"]["records"])

    def test_longitudinal_change_score_and_ancova_require_paired_subjects(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
                longitudinal_enabled=True,
                longitudinal_require_pairs=False,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_longitudinal_detail_analysis(
            bundle,
            analysis_result,
            "roi_a|roi_b",
            config=LongitudinalDetailConfig(model_family="change_score", baseline_value="M1", followup_value="M2"),
            regression_covariates=["age"],
        )
        self.assertIn("change_score", result)
        self.assertIsNone(result["change_score"].get("message"))
        self.assertIsNone(result["ancova"].get("message"))

    def test_patterns_feature_data_builds_subject_by_edge_matrix(self) -> None:
        bundle = make_network_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        feature_data = build_patterns_feature_data(bundle, analysis_result, PatternsAnalysisConfig())
        self.assertEqual(feature_data["matrix"].shape[0], 4)
        self.assertEqual(feature_data["matrix"].shape[1], 6)
        self.assertIn("age", feature_data["metadata"].columns)

    def test_patterns_analysis_returns_embedding_clustering_feature_and_pls_outputs(self) -> None:
        bundle = make_network_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_patterns_analysis(
            bundle,
            analysis_result,
            PatternsAnalysisConfig(
                mode="embedding",
                embedding_method="pca",
                cluster_method="hierarchical",
                feature_pattern_level="edges",
                cca_pls_method="pls",
                behavior_variables=("age", "score"),
                n_components=2,
            ),
        )
        self.assertEqual(result["embedding_results"]["method"], "pca")
        self.assertEqual(len(result["embedding_results"]["metadata"]), 4)
        self.assertEqual(result["subject_cluster_results"]["method"], "hierarchical")
        self.assertEqual(result["feature_pattern_results"]["level"], "edges")
        self.assertEqual(result["brain_behavior_results"]["method"], "pls")
        self.assertEqual(result["brain_behavior_results"]["behavior_variables"], ["age", "score"])

    def test_longitudinal_delta_analysis_pairs_by_idx(self) -> None:
        config = AnalysisConfig(
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=self.bundle.frequencies[18],
            freq_max=self.bundle.frequencies[20],
            mtime_filter="All",
            correction_method="none",
            longitudinal_enabled=True,
            longitudinal_column="mtime",
            longitudinal_value_a="M1",
            longitudinal_value_b="M2",
        )
        result = run_analysis(self.bundle, config)
        self.assertTrue(result["longitudinal_enabled"])
        self.assertEqual(result["longitudinal_column"], "mtime")
        self.assertEqual(len(result["pair_results"]), 45)
        self.assertTrue(any(pair["n"] >= 7 for pair in result["pair_results"]))
        first_detail = result["pair_results"][0]["detail_records"][0]
        self.assertIn("delta_a", first_detail)
        self.assertIn("delta_b", first_detail)
        self.assertIn("base_subject_id", first_detail)

    def test_longitudinal_analysis_can_switch_between_paired_and_all_subjects(self) -> None:
        bundle = DatasetBundle(
            dataset_id="synthetic",
            metric="conn_coh",
            channels=["roi_a", "roi_b"],
            trial_ids=[1, 2],
            frequencies=[10.0],
            subjects=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "subject_id": "S1_M1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1"},
                    {"dataset_id": "synthetic", "subject_id": "S1_M2", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M2"},
                    {"dataset_id": "synthetic", "subject_id": "S2_M1", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1"},
                    {"dataset_id": "synthetic", "subject_id": "S2_M2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M2"},
                    {"dataset_id": "synthetic", "subject_id": "S3_M2", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M2"},
                ]
            ),
            connectivity=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S1_M1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S1_M1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 1.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S1_M2", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M2", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S1_M2", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M2", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S2_M1", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S2_M1", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S2_M2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M2", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S2_M2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M2", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 3.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S3_M2", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M2", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "S3_M2", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M2", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 4.0},
                ]
            ),
        )
        paired = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="A",
                longitudinal_enabled=True,
                longitudinal_require_pairs=True,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        all_subjects = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="A",
                longitudinal_enabled=True,
                longitudinal_require_pairs=False,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        paired_pair = paired["pair_results"][0]
        all_pair = all_subjects["pair_results"][0]
        self.assertEqual(paired_pair["n"], 2)
        self.assertEqual(all_pair["n"], 5)
        self.assertEqual([entry["group_label"] for entry in all_pair["group_stats"]], ["M1", "M2"])
        self.assertEqual(all_subjects["test_label"], "Welch t-test on trial deltas between timepoints")

    def test_between_group_trial_comparison_works_and_same_group_falls_back(self) -> None:
        bundle = DatasetBundle(
            dataset_id="synthetic",
            metric="conn_coh",
            channels=["roi_a", "roi_b"],
            trial_ids=[1, 2],
            frequencies=[10.0],
            subjects=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1"},
                    {"dataset_id": "synthetic", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1"},
                    {"dataset_id": "synthetic", "subject_id": "B1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1"},
                    {"dataset_id": "synthetic", "subject_id": "B2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1"},
                ]
            ),
            connectivity=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 1.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 4.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 1.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B1", "idx": "3", "base_subject_id": "3", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 6.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B2", "idx": "4", "base_subject_id": "4", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 8.0},
                ]
            ),
        )

        same_group = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="A",
            ),
        )
        between_groups = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )

        self.assertFalse(same_group["between_groups"])
        self.assertTrue(between_groups["between_groups"])
        self.assertEqual(between_groups["selected_group_a"], "A")
        self.assertEqual(between_groups["selected_group_b"], "B")
        pair = between_groups["pair_results"][0]
        self.assertAlmostEqual(pair["mean_delta"], 4.0)
        self.assertEqual(pair["n_group_a"], 2)
        self.assertEqual(pair["n_group_b"], 2)
        self.assertEqual(len(pair["group_stats"]), 2)
        self.assertEqual(pair["group_stats"][0]["group_label"], "A")
        self.assertEqual(pair["group_stats"][1]["group_label"], "B")
        self.assertAlmostEqual(pair["group_stats"][0]["mean"], 1.5)
        self.assertAlmostEqual(pair["group_stats"][1]["mean"], 5.5)

        pooled_all = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a=ALL_GROUPS_VALUE,
                group_b=ALL_GROUPS_VALUE,
            ),
        )
        pooled_pair = pooled_all["pair_results"][0]
        self.assertFalse(pooled_all["between_groups"])
        self.assertEqual(pooled_all["selected_group_a"], ALL_GROUPS_LABEL)
        self.assertEqual(pooled_all["selected_group_b"], ALL_GROUPS_LABEL)
        self.assertEqual(pooled_pair["n"], 4)
        self.assertEqual(len(pooled_pair["group_stats"]), 1)
        self.assertEqual(pooled_pair["group_stats"][0]["group_label"], ALL_GROUPS_LABEL)
        self.assertAlmostEqual(pooled_pair["group_stats"][0]["mean"], 3.5)

    def test_covariate_analysis_reports_groupwise_correlations_and_comparison(self) -> None:
        bundle = DatasetBundle(
            dataset_id="synthetic",
            metric="conn_coh",
            channels=["roi_a", "roi_b"],
            trial_ids=[1, 2],
            frequencies=[10.0],
            subjects=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "age": 10.0},
                    {"dataset_id": "synthetic", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "age": 20.0},
                    {"dataset_id": "synthetic", "subject_id": "A3", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M1", "age": 30.0},
                    {"dataset_id": "synthetic", "subject_id": "A4", "idx": "4", "base_subject_id": "4", "group_label": "A", "mtime": "M1", "age": 40.0},
                    {"dataset_id": "synthetic", "subject_id": "B1", "idx": "5", "base_subject_id": "5", "group_label": "B", "mtime": "M1", "age": 10.0},
                    {"dataset_id": "synthetic", "subject_id": "B2", "idx": "6", "base_subject_id": "6", "group_label": "B", "mtime": "M1", "age": 20.0},
                    {"dataset_id": "synthetic", "subject_id": "B3", "idx": "7", "base_subject_id": "7", "group_label": "B", "mtime": "M1", "age": 30.0},
                    {"dataset_id": "synthetic", "subject_id": "B4", "idx": "8", "base_subject_id": "8", "group_label": "B", "mtime": "M1", "age": 40.0},
                ]
            ),
            connectivity=pd.DataFrame(
                [
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A1", "idx": "1", "base_subject_id": "1", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 1.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A2", "idx": "2", "base_subject_id": "2", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A3", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A3", "idx": "3", "base_subject_id": "3", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 3.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A4", "idx": "4", "base_subject_id": "4", "group_label": "A", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "A4", "idx": "4", "base_subject_id": "4", "group_label": "A", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 4.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B1", "idx": "5", "base_subject_id": "5", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B1", "idx": "5", "base_subject_id": "5", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 4.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B2", "idx": "6", "base_subject_id": "6", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B2", "idx": "6", "base_subject_id": "6", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 3.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B3", "idx": "7", "base_subject_id": "7", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B3", "idx": "7", "base_subject_id": "7", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 2.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B4", "idx": "8", "base_subject_id": "8", "group_label": "B", "mtime": "M1", "trial_id": 1, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 0.0},
                    {"dataset_id": "synthetic", "metric": "conn_coh", "subject_id": "B4", "idx": "8", "base_subject_id": "8", "group_label": "B", "mtime": "M1", "trial_id": 2, "freq": 10.0, "roi_from": "roi_a", "roi_to": "roi_b", "value": 1.0},
                ]
            ),
        )
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )

        covariate_result = run_covariate_analysis(
            bundle,
            analysis_result,
            "roi_a|roi_b",
            correlation_method="pearson",
            correlation_variable="age",
            regression_covariates=None,
        )

        self.assertIn("correlation", covariate_result)
        self.assertIn("correlation_by_group", covariate_result)
        self.assertEqual(len(covariate_result["correlation_by_group"]), 2)
        self.assertAlmostEqual(covariate_result["correlation_by_group"][0]["statistic"], 1.0)
        self.assertAlmostEqual(covariate_result["correlation_by_group"][1]["statistic"], -1.0)
        self.assertIn("correlation_group_comparison", covariate_result)
        comparison = covariate_result["correlation_group_comparison"]
        self.assertEqual(comparison["group_a"], "A")
        self.assertEqual(comparison["group_b"], "B")
        self.assertLess(comparison["p_value"], 0.05)

    def test_multivariate_regression_handles_standard_trial_one_group(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a=ALL_GROUPS_VALUE,
                group_b=ALL_GROUPS_VALUE,
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["response_definition"], "Subject-level trial delta (Trial B - Trial A)")
        self.assertEqual(result["observation_unit"], "One row per subject")
        self.assertIsNone(result["primary_effect"])
        self.assertEqual(result["n"], 6)

    def test_multivariate_regression_handles_standard_trial_two_groups(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["design_effect_name"], "group_indicator")
        self.assertEqual(result["primary_effect"]["name"], "group_indicator")
        self.assertEqual(result["n"], 6)

    def test_multivariate_regression_handles_longitudinal_paired_one_group(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="A",
                longitudinal_enabled=True,
                longitudinal_require_pairs=True,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["response_definition"], "Subject-level paired longitudinal delta difference")
        self.assertEqual(result["n"], 3)

    def test_multivariate_regression_handles_longitudinal_paired_two_groups(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
                longitudinal_enabled=True,
                longitudinal_require_pairs=True,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["design_effect_name"], "group_indicator")
        self.assertEqual(result["primary_effect"]["name"], "group_indicator")
        self.assertEqual(result["n"], 5)

    def test_multivariate_regression_handles_longitudinal_all_subjects_one_group(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="A",
                longitudinal_enabled=True,
                longitudinal_require_pairs=False,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["response_definition"], "Timepoint-level trial delta")
        self.assertEqual(result["design_effect_name"], "timepoint_indicator")
        self.assertEqual(result["primary_effect"]["name"], "timepoint_indicator")
        self.assertEqual(result["n"], 7)

    def test_multivariate_regression_handles_longitudinal_all_subjects_two_groups_with_interaction(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                correction_method="none",
                group_a="A",
                group_b="B",
                longitudinal_enabled=True,
                longitudinal_require_pairs=False,
                longitudinal_column="mtime",
                longitudinal_value_a="M1",
                longitudinal_value_b="M2",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age"])
        self.assertIsNone(result["message"])
        self.assertEqual(result["design_effect_name"], "interaction")
        self.assertEqual(result["primary_effect"]["name"], "interaction")
        self.assertTrue(any(row["name"] == "interaction" for row in result["coefficients"]))

    def test_multivariate_regression_allows_subject_variable_outcome_and_delta_covariate(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_multivariate_regression_analysis(
            bundle,
            analysis_result,
            "roi_a|roi_b",
            outcome_variable="age",
            regression_covariates=[SELECTED_EDGE_DELTA_VALUE],
        )
        self.assertIsNone(result["message"])
        self.assertEqual(result["response_definition"], "age")
        self.assertEqual(result["n"], 6)
        self.assertTrue(any(row["name"] == SELECTED_EDGE_DELTA_VALUE for row in result["coefficients"]))

    def test_multivariate_regression_reports_insufficient_data_when_covariates_overfit(self) -> None:
        bundle = make_regression_test_bundle()
        analysis_result = run_analysis(
            bundle,
            AnalysisConfig(
                dataset_id="synthetic",
                metric="conn_coh",
                trial_a=1,
                trial_b=2,
                freq_min=10.0,
                freq_max=10.0,
                mtime_filter="M1",
                correction_method="none",
                group_a="A",
                group_b="B",
            ),
        )
        result = run_multivariate_regression_analysis(bundle, analysis_result, "roi_a|roi_b", regression_covariates=["age", "score", "bmi", "weight"])
        self.assertIn("message", result)
        self.assertEqual(result["message"], "Not enough complete observations for regression.")


if __name__ == "__main__":
    unittest.main()
