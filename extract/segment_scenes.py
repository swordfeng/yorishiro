"""Segment scenes within a chapter file.

Usage:
    uv run python -m extract.segment_scenes material/processed/novel/CPK/chapters/ch000.txt [material/processed/novel/CPK/scenes/ch000/]

Input:
    YAML frontmatter chapter file from chapters_epub.py

Output:
    Individual scene files in the output directory:
    ├── scene_001.txt
    ├── scene_002.txt
    └── ...

File format (YAML frontmatter):
    ---
    scene_id: "c000_s001"
    chapter_index: 0
    chapter_title: "第一章 降临"
    boundary_type: "location_change"
    boundary_reason: "从月读空间切换到地球某城市"
    location: "地球-某城市-街头"
    time_of_day: "傍晚"
    characters_present: ["辉夜"]
    characters_mentioned: []
    pov_character: "辉夜"
    ---
    [scene text content...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml


@dataclass
class Scene:
    """Represents a scene within a chapter."""
    scene_id: str
    chapter_index: int
    chapter_title: str
    content: str
    boundary_type: str
    boundary_reason: str
    location: str = ""
    time_of_day: str = ""
    characters_present: list[str] = None
    characters_mentioned: list[str] = None
    pov_character: str = ""
    
    def __post_init__(self):
        if self.characters_present is None:
            self.characters_present = []
        if self.characters_mentioned is None:
            self.characters_mentioned = []


def segment_scenes(
    chapter_file: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """Segment a chapter into scenes and save as YAML frontmatter files.
    
    Args:
        chapter_file: Path to the chapter file (YAML frontmatter format).
        output_dir: Directory to save scene files. Defaults to scenes/<chapter_id>/.
        
    Returns:
        List of paths to saved scene files.
    """
    if not chapter_file.exists():
        raise FileNotFoundError(f"Chapter file not found: {chapter_file}")
    
    # Parse chapter file
    chapter_metadata, chapter_content = parse_frontmatter(chapter_file)
    
    chapter_index = chapter_metadata.get("index", 0)
    chapter_title = chapter_metadata.get("title", f"Chapter {chapter_index}")
    
    # Determine output directory
    if output_dir is None:
        chapter_id = f"ch{chapter_index:03d}"
        output_dir = chapter_file.parent.parent / "scenes" / chapter_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Segment scenes (placeholder - will use LLM in real implementation)
    scenes = segment_with_llm(chapter_content, chapter_index, chapter_title)
    
    # Save each scene
    saved_files = []
    for i, scene in enumerate(scenes, 1):
        scene_path = save_scene(scene, output_dir)
        saved_files.append(scene_path)
    
    return saved_files


def parse_frontmatter(file_path: Path) -> tuple[dict, str]:
    """Parse a YAML frontmatter file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata, body
    
    return {}, content


def segment_with_llm(
    chapter_content: str,
    chapter_index: int,
    chapter_title: str,
) -> list[Scene]:
    """Segment chapter into scenes using LLM.
    
    This is a placeholder. Real implementation will use PydanticAI.
    For now, creates a single scene for the entire chapter.
    """
    # Placeholder: return entire chapter as one scene
    return [
        Scene(
            scene_id=f"c{chapter_index:03d}_s001",
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            content=chapter_content,
            boundary_type="chapter_start",
            boundary_reason="章节开始",
            location="N/A",
            time_of_day="N/A",
        )
    ]


def save_scene(scene: Scene, output_dir: Path) -> Path:
    """Save a scene to a YAML frontmatter file."""
    scene_num = int(scene.scene_id.split('_s')[-1])
    filename = f"scene_{scene_num:03d}.txt"
    output_path = output_dir / filename
    
    frontmatter = {
        "scene_id": scene.scene_id,
        "chapter_index": scene.chapter_index,
        "chapter_title": scene.chapter_title,
        "boundary_type": scene.boundary_type,
        "boundary_reason": scene.boundary_reason,
        "location": scene.location or "N/A",
        "time_of_day": scene.time_of_day or "N/A",
        "characters_present": scene.characters_present or [],
        "characters_mentioned": scene.characters_mentioned or [],
        "pov_character": scene.pov_character or "",
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, allow_unicode=True, sort_keys=False)
        f.write("---\n")
        f.write(scene.content)
    
    return output_path


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m extract.segment_scenes <chapter_file> [output_folder_path]")
        print("")
        print("Examples:")
        print("  python -m extract.segment_scenes ch000.txt")
        print("  python -m extract.segment_scenes ch000.txt scenes/ch000/")
        sys.exit(1)
    
    chapter_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        # Default: scenes/<chapter_id>/
        output_dir = None
    
    try:
        saved_files = segment_scenes(chapter_file, output_dir)
        
        print(f"Segmented {chapter_file.name} into {len(saved_files)} scenes")
        print(f"Output: {saved_files[0].parent if saved_files else 'N/A'}")
        for path in saved_files[:5]:
            print(f"  - {path.name}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
