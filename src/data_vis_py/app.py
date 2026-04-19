"""Application entrypoint for the connectivity dashboard."""

from __future__ import annotations

from pathlib import Path

from data_vis_py.io.dataset_loader import list_dataset_files, list_datasets, load_dataset
from data_vis_py.ui.dashboard import create_dashboard


def create_app():
    """Create and return the Dash app instance."""
    repo_root = Path(__file__).resolve().parents[2]
    raw_root = repo_root / "data" / "raw"
    dataset_ids = list_datasets(raw_root)
    if not dataset_ids:
        raise FileNotFoundError(f"No datasets with JSON and CSV files found in {raw_root}")
    initial_dataset = "REST_24_Stroke" if "REST_24_Stroke" in dataset_ids else dataset_ids[0]
    dataset_dir = raw_root / initial_dataset
    files = list_dataset_files(dataset_dir)
    initial_json = "data_coh.json" if "data_coh.json" in files["json_files"] else files["json_files"][0]
    initial_csv = "info.csv" if "info.csv" in files["csv_files"] else files["csv_files"][0]
    bundle = load_dataset(dataset_dir, json_filename=initial_json, csv_filename=initial_csv)
    return create_dashboard(
        raw_root=raw_root,
        dataset_ids=dataset_ids,
        initial_dataset=initial_dataset,
        initial_bundle=bundle,
        json_files=files["json_files"],
        csv_files=files["csv_files"],
        initial_json=initial_json,
        initial_csv=initial_csv,
    )


def main() -> None:
    """Run the local development server."""
    app = create_app()
    app.run(debug=False)


if __name__ == "__main__":
    main()
