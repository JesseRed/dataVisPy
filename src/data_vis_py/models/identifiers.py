"""Helpers for deriving stable identifiers from dataset-specific subject IDs."""

from __future__ import annotations


def derive_base_subject_id(subject_id: str) -> str:
    """Return a subject-level identifier shared by repeated measurements."""
    parts = subject_id.split("_")
    if parts and parts[-1] == "merged":
        parts = parts[:-1]
    if len(parts) < 2:
        return subject_id
    return "_".join(parts[:-1])


def normalize_mtime(value: object) -> str:
    """Normalize measurement time labels to a stable `M<number>` format."""
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "Unknown"
    if text.upper().startswith("M"):
        return text.upper()
    try:
        numeric = int(float(text))
    except ValueError:
        return text
    return f"M{numeric}"
