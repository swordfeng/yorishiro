"""Epub extraction pipeline for Yorishiro."""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterator


@dataclass
class Chapter:
    """Represents a chapter from an epub."""
    index: int
    title: str
    content: str
    path: str = ""


@dataclass
class ExtractedEpub:
    """Represents a fully extracted epub."""
    title: str
    author: str
    language: str
    chapters: list[Chapter] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)


def parse_epub(epub_path: str | Path) -> ExtractedEpub:
    """Parse an epub file and extract its contents.

    Args:
        epub_path: Path to the epub file.

    Returns:
        ExtractedEpub with all chapters and metadata.
    """
    import ebooklib
    from ebooklib import epub

    path = Path(epub_path)
    book = epub.read_epub(str(path))

    metadata = {}
    if book.metadata and book.metadata.get('0'):
        meta = book.metadata['0']
        metadata = {
            'title': _get_metadata_value(meta, 'title'),
            'creator': _get_metadata_value(meta, 'creator'),
            'language': _get_metadata_value(meta, 'language'),
        }

    chapters = []
    items = list(book.get_items())
    chapter_count = 0

    for item in items:
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            text = _extract_text_from_html(content)
            if text.strip():
                title = _extract_title_from_html(content) or f"Chapter {chapter_count + 1}"
                item_path = getattr(item, 'href', None) or getattr(item, 'file_name', str(item.get_name()))
                chapters.append(Chapter(
                    index=chapter_count,
                    title=title,
                    content=text,
                    path=item_path
                ))
                chapter_count += 1

    return ExtractedEpub(
        title=metadata.get('title', path.stem),
        author=metadata.get('creator', 'Unknown'),
        language=metadata.get('language', 'unknown'),
        chapters=chapters,
        raw_metadata=metadata
    )


def _get_metadata_value(meta: dict, key: str) -> str:
    """Extract metadata value from Calibre metadata format."""
    if key in meta and meta[key]:
        values = meta[key]
        if isinstance(values, list) and values:
            return str(values[0][0]) if isinstance(values[0], tuple) else str(values[0])
        elif isinstance(values, list):
            return str(values[0])
        return str(values)
    return ''


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags and extract plain text."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_title_from_html(html: str) -> str | None:
    """Extract title from HTML content (h1, title, or first strong text)."""
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
    if title_match:
        return _extract_text_from_html(title_match.group(1)).strip()

    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()

    strong_match = re.search(r'<strong[^>]*>(.*?)</strong>', html, re.DOTALL | re.IGNORECASE)
    if strong_match:
        text = _extract_text_from_html(strong_match.group(1)).strip()
        if text and len(text) < 100:
            return text

    return None


def extract_to_json(epub_path: str | Path, output_path: str | Path | None = None) -> dict:
    """Parse epub and save to JSON for inspection.

    Args:
        epub_path: Path to epub file.
        output_path: Optional path to save JSON output.

    Returns:
        Dictionary representation of extracted epub.
    """
    extracted = parse_epub(epub_path)
    data = asdict(extracted)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def iter_chapters(epub_path: str | Path) -> Iterator[Chapter]:
    """Iterate through chapters without loading everything into memory.

    Args:
        epub_path: Path to epub file.

    Yields:
        Chapter objects one at a time.
    """
    extracted = parse_epub(epub_path)
    for chapter in extracted.chapters:
        yield chapter


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m extract.epub_pipeline <epub_path> [output_json_path]")
        sys.exit(1)

    epub_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    data = extract_to_json(epub_path, output_path)
    print(f"Extracted: {data['title']} by {data['author']}")
    print(f"Chapters: {len(data['chapters'])}")
    if output_path:
        print(f"Saved to: {output_path}")
