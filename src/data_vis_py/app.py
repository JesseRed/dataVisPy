"""Application entrypoint for the connectivity dashboard."""

from __future__ import annotations

from pathlib import Path

from data_vis_py.io.dataset_loader import list_dataset_files, load_dataset
from data_vis_py.ui.dashboard import create_dashboard


def create_app():
    """Create and return the Dash app instance."""
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "data" / "raw" / "REST_24_Stroke"
    files = list_dataset_files(dataset_dir)
    initial_json = "data_coh.json" if "data_coh.json" in files["json_files"] else files["json_files"][0]
    initial_csv = "info.csv" if "info.csv" in files["csv_files"] else files["csv_files"][0]
    bundle = load_dataset(dataset_dir, json_filename=initial_json, csv_filename=initial_csv)
    return create_dashboard(
        dataset_dir=dataset_dir,
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
