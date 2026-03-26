"""Shared utility helpers for the goalkeeper project."""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
