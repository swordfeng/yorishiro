"""Extract chapters from EPUB to YAML frontmatter files.

Usage:
    uv run python3 extract_chapters.py --input material/raw/CPK.epub --output material/processed/novel/CPK/chapters

Output format:
    Each chapter is saved as chXXX.txt with YAML frontmatter:
    ---
    index: 0
    title: "第一章 降临"
    length: 15234
    source_file: "CPK.epub"
    ---
    [chapter content]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml


@dataclass
class Chapter:
    """Represents a chapter from an epub."""
    index: int
    title: str
    content: str
    path: str = ""


def parse_epub(epub_path: Path) -> Iterator[Chapter]:
    """Parse an epub file and yield chapters.
    
    Args:
        epub_path: Path to the epub file.
        
    Yields:
        Chapter objects.
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(epub_path))
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html_content = item.get_content().decode('utf-8', errors='ignore')
            text = _extract_text_from_html(html_content)
            
            if text.strip():
                title = _extract_title_from_html(html_content) or f"Chapter {chapter_count + 1}"
                item_path = getattr(item, 'href', None) or getattr(item, 'file_name', str(item.get_name()))
                
                yield Chapter(
                    index=chapter_count,
                    title=title,
                    content=text,
                    path=item_path
                )
                chapter_count += 1


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags and extract plain text."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_title_from_html(html: str) -> str | None:
    """Extract title from HTML content."""
    # Try h1 first
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
    if title_match:
        return _extract_text_from_html(title_match.group(1)).strip()

    # Try title tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()

    # Try strong tag
    strong_match = re.search(r'<strong[^>]*>(.*?)</strong>', html, re.DOTALL | re.IGNORECASE)
    if strong_match:
        text = _extract_text_from_html(strong_match.group(1)).strip()
        if text and len(text) < 100:
            return text

    return None


def save_chapter_to_yaml(chapter: Chapter, output_dir: Path, source_file: str) -> Path:
    """Save a chapter to a YAML frontmatter file.
    
    Args:
        chapter: The chapter to save.
        output_dir: Directory to save the file.
        source_file: Name of the source epub file (for metadata).
        
    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename: chXXX.txt
    filename = f"ch{chapter.index:03d}.txt"
    output_path = output_dir / filename
    
    # Build frontmatter
    frontmatter = {
        "index": chapter.index,
        "title": chapter.title,
        "length": len(chapter.content),
        "source_file": source_file,
        "path": chapter.path
    }
    
    # Write YAML frontmatter + content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, allow_unicode=True, sort_keys=False)
        f.write("---\n")
        f.write(chapter.content)
    
    return output_path


def extract_chapters(
    input_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Extract chapters from EPUB and save as YAML frontmatter files.
    
    This is the main function that can be called programmatically.
    
    Args:
        input_path: Path to the EPUB file.
        output_dir: Directory to save chapter files.
        
    Returns:
        List of paths to saved chapter files.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    source_file = input_path.name
    saved_files = []
    
    for chapter in parse_epub(input_path):
        output_path = save_chapter_to_yaml(chapter, output_dir, source_file)
        saved_files.append(output_path)
    
    return saved_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract chapters from EPUB to YAML frontmatter files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python3 extract_chapters.py -i book.epub -o chapters/
    uv run python3 extract_chapters.py --input CPK.epub --output material/processed/novel/CPK/chapters
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input EPUB file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for chapter files"
    )
    
    args = parser.parse_args()
    
    try:
        saved_files = extract_chapters(args.input, args.output)
        
        print(f"Extracted {len(saved_files)} chapters from {args.input.name}")
        print(f"Output directory: {args.output}")
        for path in saved_files[:5]:  # Show first 5
            print(f"  - {path.name}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error extracting chapters: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
