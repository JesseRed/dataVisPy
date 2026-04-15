"""Load connectivity datasets from raw JSON/CSV files and cache them as Parquet."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_vis_py.models.identifiers import derive_base_subject_id, normalize_mtime


CACHE_VERSION = 2
REQUIRED_SUBJECT_COLUMNS = ["ID", "Group", "IDX", "MTime"]


@dataclass(frozen=True)
class DatasetBundle:
    """In-memory representation of a cached dataset."""

    dataset_id: str
    metric: str
    connectivity: pd.DataFrame
    subjects: pd.DataFrame
    channels: list[str]
    trial_ids: list[int]
    frequencies: list[float]


def load_dataset(
    dataset_dir: Path | str,
    json_filename: str | None = None,
    csv_filename: str | None = None,
    cache_root: Path | str | None = None,
) -> DatasetBundle:
    """Load a dataset from cache or raw files."""
    dataset_path = Path(dataset_dir)
    selected_json = json_filename or _default_json_filename(dataset_path)
    selected_csv = csv_filename or "info.csv"
    cache_root_path = Path(cache_root) if cache_root else dataset_path.parents[1] / "processed"
    cache_dir = cache_root_path / dataset_path.name / _cache_key(selected_json, selected_csv)
    connectivity_cache = cache_dir / "connectivity.parquet"
    subjects_cache = cache_dir / "subjects.parquet"
    metadata_cache = cache_dir / "metadata.json"

    if connectivity_cache.exists() and subjects_cache.exists() and metadata_cache.exists():
        metadata = json.loads(metadata_cache.read_text())
        if (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("json_filename") == selected_json
            and metadata.get("csv_filename") == selected_csv
        ):
            connectivity = pd.read_parquet(connectivity_cache)
            subjects = pd.read_parquet(subjects_cache)
            return DatasetBundle(
                dataset_id=metadata["dataset_id"],
                metric=metadata["metric"],
                connectivity=connectivity,
                subjects=subjects,
                channels=metadata["channels"],
                trial_ids=metadata["trial_ids"],
                frequencies=metadata["frequencies"],
            )

    bundle = _load_from_raw(dataset_path, selected_json, selected_csv)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle.connectivity.to_parquet(connectivity_cache, index=False)
    bundle.subjects.to_parquet(subjects_cache, index=False)
    metadata_cache.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "dataset_id": bundle.dataset_id,
                "metric": bundle.metric,
                "json_filename": selected_json,
                "csv_filename": selected_csv,
                "channels": bundle.channels,
                "trial_ids": bundle.trial_ids,
                "frequencies": bundle.frequencies,
            },
            indent=2,
        )
    )
    return bundle


def list_dataset_files(dataset_dir: Path | str) -> dict[str, list[str]]:
    """List selectable JSON and CSV files in the dataset directory."""
    dataset_path = Path(dataset_dir)
    return {
        "json_files": sorted(path.name for path in dataset_path.glob("*.json")),
        "csv_files": sorted(path.name for path in dataset_path.glob("*.csv")),
    }


def _load_from_raw(dataset_path: Path, json_filename: str, csv_filename: str) -> DatasetBundle:
    json_path = dataset_path / json_filename
    csv_path = dataset_path / csv_filename
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw = json.loads(json_path.read_text())
    subjects = _load_subjects_table(csv_path)
    connectivity = _load_connectivity_table(raw, dataset_path.name)
    if "IDX" not in subjects.columns:
        raise ValueError("Behavior table is missing required column 'IDX'.")

    idx_values = subjects["IDX"].astype(str).str.strip()
    base_subject_ids = idx_values.where(idx_values != "", subjects["ID"].map(derive_base_subject_id))

    subjects = subjects.assign(
        dataset_id=dataset_path.name,
        idx=idx_values,
        base_subject_id=base_subject_ids,
        group_label=subjects["Group"].astype(str),
        mtime=subjects["MTime"].map(normalize_mtime),
        subject_id=subjects["ID"],
    ).rename(columns={"ID": "subject_id_raw"})

    subject_columns = ["dataset_id", "subject_id", "idx", "base_subject_id", "group_label", "mtime"]
    remaining_columns = [
        column
        for column in subjects.columns
        if column not in {"subject_id_raw", "Group", "IDX", "MTime"}
        and column not in subject_columns
    ]
    subjects = subjects[subject_columns + remaining_columns].sort_values("subject_id").reset_index(drop=True)

    connectivity = connectivity.merge(
        subjects[["subject_id", "idx", "base_subject_id", "group_label", "mtime"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    return DatasetBundle(
        dataset_id=dataset_path.name,
        metric=str(raw.get("type", "unknown")),
        connectivity=connectivity,
        subjects=subjects,
        channels=list(raw["channels"]),
        trial_ids=[int(trial_id) for trial_id in raw["trials"]],
        frequencies=[float(freq) for freq in raw["freq"]],
    )


def _default_json_filename(dataset_path: Path) -> str:
    for candidate in ["data_coh.json", "export_coh.json"]:
        if (dataset_path / candidate).exists():
            return candidate
    json_files = sorted(path.name for path in dataset_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")
    return json_files[0]


def _cache_key(json_filename: str, csv_filename: str) -> str:
    combined = f"{json_filename}__{csv_filename}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", combined)


def _load_subjects_table(csv_path: Path) -> pd.DataFrame:
    sample = csv_path.read_text(encoding="utf-8")[:2048]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    decimal = "," if delimiter == ";" else "."
    subjects = pd.read_csv(csv_path, sep=delimiter, decimal=decimal, na_values=["NA"])
    missing = [column for column in REQUIRED_SUBJECT_COLUMNS if column not in subjects.columns]
    if missing:
        raise ValueError(f"Behavior table is missing required columns: {', '.join(missing)}")
    return subjects


def _load_connectivity_table(raw: dict[str, Any], dataset_id: str) -> pd.DataFrame:
    roi_from = raw["channelcmb"]["from"]
    roi_to = raw["channelcmb"]["to"]
    frequencies = [float(freq) for freq in raw["freq"]]
    rows: list[dict[str, Any]] = []
    metric = str(raw.get("type", "unknown"))

    for subject in raw["subjects"]:
        subject_id = subject["subject_id"]
        for trial in subject["trials"]:
            trial_id = int(trial["trial_id"])
            for pair_index, spectrum in enumerate(trial["dat"]):
                source = roi_from[pair_index]
                target = roi_to[pair_index]
                for freq_index, value in enumerate(spectrum):
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "metric": metric,
                            "subject_id": subject_id,
                            "trial_id": trial_id,
                            "freq": frequencies[freq_index],
                            "roi_from": source,
                            "roi_to": target,
                            "value": float(value),
                        }
                    )

    connectivity = pd.DataFrame.from_records(rows)
    connectivity["trial_id"] = connectivity["trial_id"].astype("int64")
    connectivity["freq"] = connectivity["freq"].astype("float64")
    connectivity["value"] = connectivity["value"].astype("float64")
    return connectivity
