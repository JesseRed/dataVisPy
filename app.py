"""Convenience runner for local development."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the Dash app while keeping the `src/` layout."""
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from data_vis_py.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
