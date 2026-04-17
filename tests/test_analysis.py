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
    SELECTED_EDGE_DELTA_VALUE,
    run_analysis,
    run_covariate_analysis,
    run_multivariate_regression_analysis,
    summarize_pair_result,
)


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
