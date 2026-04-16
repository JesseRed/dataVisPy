"""Tests for the core connectivity analysis routines."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_vis_py.io.dataset_loader import DatasetBundle, load_dataset
from data_vis_py.stats.analysis import AnalysisConfig, run_analysis, run_covariate_analysis, summarize_pair_result


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
            analysis_mode="trial_delta",
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
            analysis_mode="trial_delta",
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

    def test_session_delta_analysis_uses_repeated_measurements(self) -> None:
        config = AnalysisConfig(
            analysis_mode="session_delta",
            dataset_id=self.bundle.dataset_id,
            metric=self.bundle.metric,
            trial_a=self.bundle.trial_ids[0],
            trial_b=self.bundle.trial_ids[1],
            freq_min=self.bundle.frequencies[25],
            freq_max=self.bundle.frequencies[25],
            mtime_filter="All",
            correction_method="fdr_bh",
        )
        result = run_analysis(self.bundle, config)
        self.assertEqual(len(result["pair_results"]), 45)
        self.assertTrue(any(pair["n"] >= 7 for pair in result["pair_results"]))

    def test_frequency_band_average_matches_manual_mean(self) -> None:
        freq_min = self.bundle.frequencies[15]
        freq_max = self.bundle.frequencies[17]
        config = AnalysisConfig(
            analysis_mode="trial_delta",
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
            analysis_mode="trial_delta",
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
            analysis_mode="trial_delta",
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
                analysis_mode="trial_delta",
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
                analysis_mode="trial_delta",
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


if __name__ == "__main__":
    unittest.main()
