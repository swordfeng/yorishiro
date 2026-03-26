"""Yorishiro extraction layer."""

from .epub_pipeline import (
    parse_epub,
    extract_to_json,
    iter_chapters,
    Chapter,
    ExtractedEpub,
)
from .character_extractor import (
    SYSTEM_PROMPT as EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
)
from .scene_segmentation import (
    SYSTEM_PROMPT as SEGMENTATION_SYSTEM_PROMPT,
    build_scene_segmentation_prompt,
)

__all__ = [
    "parse_epub",
    "extract_to_json",
    "iter_chapters",
    "Chapter",
    "ExtractedEpub",
    "EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_prompt",
    "SEGMENTATION_SYSTEM_PROMPT",
    "build_scene_segmentation_prompt",
]
