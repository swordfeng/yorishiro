"""Split chapters.json into individual chapter files."""

import json
from pathlib import Path


def split_chapters(
    chapters_file: str | Path,
    output_dir: str | Path,
) -> None:
    """Split combined chapters.json into individual chapter files.

    Args:
        chapters_file: Path to chapters.json
        output_dir: Directory to save individual chapter files
    """
    chapters_file = Path(chapters_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(chapters_file, encoding="utf-8") as f:
        data = json.load(f)

    metadata = {
        "title": data["title"],
        "author": data["author"],
        "language": data["language"],
        "chapter_count": data["chapter_count"],
    }

    for ch in data["chapters"]:
        chapter_file = output_dir / f"chapter_{ch['index']:03d}.json"
        chapter_data = {
            **metadata,
            "chapter_index": ch["index"],
            "title": ch["title"],
            "content": ch["content"],
            "path": ch["path"],
            "word_count": ch["word_count"],
        }

        with open(chapter_file, "w", encoding="utf-8") as f:
            json.dump(chapter_data, f, ensure_ascii=False, indent=2)

        print(f"Created: {chapter_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split chapters.json into files")
    parser.add_argument(
        "--input",
        default="material/processed/novel/CPK/chapters/chapters.json",
        help="Input chapters.json",
    )
    parser.add_argument(
        "--output",
        default="material/processed/novel/CPK/chapters",
        help="Output directory",
    )

    args = parser.parse_args()
    split_chapters(args.input, args.output)
