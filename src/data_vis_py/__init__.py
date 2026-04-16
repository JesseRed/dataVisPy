"""Interactive connectivity analysis for MEG datasets."""

__all__ = ["main"]


def main() -> None:
    """Run the dashboard entrypoint without importing UI dependencies at package import time."""
    from .app import main as app_main

    app_main()
