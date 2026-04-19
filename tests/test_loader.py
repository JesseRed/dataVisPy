"""Tests for dataset loading and normalization."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_vis_py.io.dataset_loader import list_datasets, load_dataset


class DatasetLoaderTests(unittest.TestCase):
    """Integration checks for raw dataset import."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.raw_root = self.repo_root / "data" / "raw"
        self.dataset_dir = self.repo_root / "data" / "raw" / "REST_24_Stroke"

    def test_lists_available_dataset_directories(self) -> None:
        dataset_ids = list_datasets(self.raw_root)

        self.assertIn("REST_24_Stroke", dataset_ids)
        self.assertIn("sleep", dataset_ids)

    def test_loads_and_normalizes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = load_dataset(self.dataset_dir, cache_root=tmp_dir)

        self.assertEqual(bundle.dataset_id, "REST_24_Stroke")
        self.assertEqual(bundle.metric, "conn_coh")
        self.assertEqual(len(bundle.subjects), 38)
        self.assertEqual(len(bundle.connectivity), 38 * 5 * 45 * 206)
        self.assertTrue({"dataset_id", "subject_id", "idx", "base_subject_id", "group_label", "mtime"}.issubset(bundle.subjects.columns))
        self.assertTrue({"dataset_id", "metric", "subject_id", "idx", "base_subject_id", "group_label", "mtime", "trial_id", "freq", "roi_from", "roi_to", "value"}.issubset(bundle.connectivity.columns))
        self.assertIn("7", set(bundle.subjects["base_subject_id"]))
        self.assertEqual(sorted(bundle.subjects["mtime"].unique().tolist()), ["M1", "M2"])
        self.assertEqual(bundle.subjects.loc[bundle.subjects["subject_id"] == "rest_24_stroke_07_2.1", "idx"].iloc[0], "7")

    def test_csv_decimal_and_na_values_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = load_dataset(self.dataset_dir, cache_root=tmp_dir)

        subject_row = bundle.subjects.loc[bundle.subjects["subject_id"] == "rest_24_stroke_24_2.1"].iloc[0]
        self.assertTrue(subject_row["EQ5D"] == 80 or float(subject_row["EQ5D"]) == 80.0)
        self.assertTrue(subject_row["DSS"] != subject_row["DSS"])

    def test_explicit_json_and_csv_selection_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = load_dataset(
                self.dataset_dir,
                json_filename="data_coh.json",
                csv_filename="info.csv",
                cache_root=tmp_dir,
            )

        self.assertEqual(bundle.dataset_id, "REST_24_Stroke")
        self.assertEqual(bundle.metric, "conn_coh")


if __name__ == "__main__":
    unittest.main()
