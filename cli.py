"""CLI for Yorishiro extraction pipelines."""

import argparse
import json
import sys
from pathlib import Path

from extract.epub_pipeline import parse_epub


def extract_chapters(epub_path: str | Path, output_dir: str | Path) -> None:
    """Extract chapters from epub and save to processed directory.

    Args:
        epub_path: Path to epub file
        output_dir: Directory to save chapters.json
    """
    epub_path = Path(epub_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = parse_epub(epub_path)

    chapters_data = {
        "title": extracted.title,
        "author": extracted.author,
        "language": extracted.language,
        "chapter_count": len(extracted.chapters),
        "chapters": [
            {
                "index": ch.index,
                "title": ch.title,
                "content": ch.content,
                "path": ch.path,
                "word_count": len(ch.content),
            }
            for ch in extracted.chapters
        ],
    }

    output_path = output_dir / "chapters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chapters_data, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(extracted.chapters)} chapters to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Yorishiro extraction CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    extract_parser = subparsers.add_parser("extract", help="Extract content from epub")
    extract_parser.add_argument("epub", help="Path to epub file")
    extract_parser.add_argument("--output", "-o", required=True, help="Output directory")

    args = parser.parse_args()

    if args.command == "extract":
        extract_chapters(args.epub, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
