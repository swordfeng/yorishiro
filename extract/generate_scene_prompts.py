"""Generate scene segmentation prompts for manual LLM processing."""

import json
from pathlib import Path

from extract.scene_segmentation import SYSTEM_PROMPT, build_scene_segmentation_prompt


def generate_scene_prompts(
    chapters_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate scene segmentation prompt files for each chapter.

    Args:
        chapters_dir: Directory containing chapter_XXX.json files
        output_dir: Directory to save prompt files
    """
    chapters_dir = Path(chapters_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chapter_files = sorted(chapters_dir.glob("chapter_*.json"))

    for chapter_file in chapter_files:
        with open(chapter_file, encoding="utf-8") as f:
            data = json.load(f)

        prompt = build_scene_segmentation_prompt(
            chapter_index=data["chapter_index"],
            chapter_title=data["title"],
            chapter_content=data["content"],
        )

        output_file = output_dir / f"prompt_ch{data['chapter_index']:03d}_scene_seg.txt"
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

    parser = argparse.ArgumentParser(description="Generate scene segmentation prompts")
    parser.add_argument(
        "--chapters",
        default="material/processed/novel/CPK/chapters",
        help="Directory containing chapter files",
    )
    parser.add_argument(
        "--output",
        default="material/processed/novel/CPK/scenes/prompts",
        help="Output directory for prompt files",
    )

    args = parser.parse_args()
    generate_scene_prompts(args.chapters, args.output)
