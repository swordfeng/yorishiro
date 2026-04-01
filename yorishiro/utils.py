"""Shared utilities for yorishiro pipelines."""

from pathlib import Path


def is_output_stale(output_path: Path, source_paths: list[Path]) -> bool:
    """Return True if output needs (re)processing.

    output_path: the representative output file to timestamp-check.
    source_paths: files that, if newer than output, trigger reprocessing.
    """
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    return any(src.stat().st_mtime > output_mtime for src in source_paths if src.exists())
