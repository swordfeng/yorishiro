"""Generate character extraction prompts for manual LLM processing.

This script generates prompt files for each chapter that can be
manually copied into an LLM interface for extraction.

Usage:
    uv run python -m extract.generate_prompts
"""

import json
from pathlib import Path

from extract.character_extractor import SYSTEM_PROMPT, build_extraction_prompt


def generate_prompts(
    chapters_file: str | Path,
    character_name: str,
    output_dir: str | Path,
) -> None:
    """Generate prompt files for each chapter.

    Args:
        chapters_file: Path to chapters.json
        character_name: Character to extract
        output_dir: Directory to save prompt files
    """
    chapters_file = Path(chapters_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(chapters_file, encoding="utf-8") as f:
        data = json.load(f)

    chapters = data["chapters"]

    for ch in chapters:
        prompt = build_extraction_prompt(
            character_name=character_name,
            chapter_index=ch["index"],
            chapter_title=ch["title"],
            chapter_content=ch["content"],
        )

        output_file = output_dir / f"prompt_ch{ch["index"]:03d}_{character_name}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("SYSTEM PROMPT\n")
            f.write("=" * 60 + "\n")
            f.write(SYSTEM_PROMPT)
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write("USER PROMPT\n")
            f.write("=" * 60 + "\n")
            f.write(prompt)

        print(f"Generated: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate character extraction prompts")
    parser.add_argument(
        "--chapters",
        default="material/processed/novel/CPK/chapters/chapters.json",
        help="Path to chapters.json",
    )
    parser.add_argument(
        "--character",
        default="彩葉",
        help="Character name to extract",
    )
    parser.add_argument(
        "--output",
        default="material/processed/novel/CPK/scenes/prompts",
        help="Output directory for prompt files",
    )

    args = parser.parse_args()

    generate_prompts(args.chapters, args.character, args.output)
