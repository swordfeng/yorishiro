"""Shared utilities for yorishiro pipelines."""

from pathlib import Path

import torch


def is_output_stale(output_path: Path, source_paths: list[Path]) -> bool:
    """Return True if output needs (re)processing.

    output_path: the representative output file to timestamp-check.
    source_paths: files that, if newer than output, trigger reprocessing.
    """
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    return any(src.stat().st_mtime > output_mtime for src in source_paths if src.exists())


def get_device() -> str:
    """Select best available device: MPS (Mac) > CUDA > CPU.

    Returns device string suitable for torch.device() or CTranslate2.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
