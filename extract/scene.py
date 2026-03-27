"""Automated LLM-based scene segmentation for novel chapters.

Usage:
    uv run python -m extract.scene <chapter_file> [output_dir]
        [--model MODEL] [--base-url URL] [--api-key-env VAR] [--force]

Example:
    uv run python -m extract.scene material/processed/novel/CPK/chapters/ch003.txt \\
                                   material/processed/novel/CPK/scenes/ch003

Input:
    YAML frontmatter chapter file produced by extract.chapters_epub

Output:
    scenes_manifest.json  -- chapter metadata + scene list with offsets
    scene_000.txt, scene_001.txt, ...  -- plain text, one scene per file

Offset guarantee:
    start_offset and end_offset are Python codepoints (len()).
    Scenes are consecutive with no gaps: scene[i].end == scene[i+1].start.
    Last scene ends at total_length == len(chapter_text).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


CHUNK_SIZE = 8000  # chars per LLM batch


SYSTEM_PROMPT = """You are a professional narrative structure analyst. Your task is to identify scene boundaries in novel chapters.

## Source Material Language

ALL metadata (location, time, characters) must use the SAME language as the source material. Detect the language from the text.

## Scene Definition

A scene is a narrative unit with:
- Same location: events in the same physical/virtual space
- Continuous time: no significant jumps
- Same POV: primarily the same character's perspective
- Coherent events: direct causal relationship

## Scene Boundary Signals (split ONLY on these)

1. Location change — character moves to a different place
2. Time jump — significant time progression (hours, days later, etc.)
3. POV change — perspective character changes
4. Narrative break — explicit separator (※, ──, ・・・ as scene break, etc.)

## Merge Rule

If adjacent content shares location + time + POV with no separator → it is ONE scene. Do NOT split without clear reason. The number of scenes does not matter; correctness does.

## Your Task

You receive a window of chapter text. The cursor is at the START of the current scene (either the chapter start or a scene that was still ongoing from the previous window).

For each scene you can identify within this window, output:
- metadata: location, time, characters, boundary_type
- end_text: the LAST 40-50 characters of that scene, verbatim from the source

If the final scene in your window extends beyond what you can see (more text follows), set end_text to empty string "" and set has_more=true.

## end_text Requirements

- Copy verbatim from the source text (exact characters, including punctuation)
- Must be unique enough to locate the scene end unambiguously
- 40-50 chars is ideal; use up to 60 if needed for uniqueness
- boundary_type describes WHY this scene ends (what comes next)

## Character Names

List names EXACTLY as they appear in this scene's text. Do not normalize or translate.

## Non-Narrative Content

For non-narrative sections (caution pages, TOC, colophon, etc.): one scene, boundary_type="non_narrative", location="N/A", time="N/A", characters=[].
"""


class SceneSegment(BaseModel):
    """One scene. Starts at cursor (or where the previous scene ended)."""

    location: str = Field(description="Scene location in source material language")
    time: str = Field(description="Time of day in source material language")
    characters: list[str] = Field(description="Character names exactly as they appear in this scene")
    boundary_type: Literal[
        "chapter_start",
        "location_change",
        "time_jump",
        "pov_change",
        "narrative_break",
        "non_narrative",
    ] = Field(description="Type of boundary that ends this scene")
    end_text: str = Field(
        description=(
            "Last 40-60 characters of this scene, verbatim from source. "
            "Empty string '' if this scene extends beyond the current text window."
        )
    )


class SegmentationResult(BaseModel):
    scenes: list[SceneSegment] = Field(
        description=(
            "Scenes in order. First scene starts at the cursor position. "
            "Each subsequent scene starts exactly where the previous one ended."
        )
    )
    has_more: bool = Field(
        description="True if the chapter continues beyond this window and the last scene is incomplete"
    )
    summary: str = Field(
        description="Brief summary of the content processed in this window, for continuity in the next call"
    )


@dataclass
class SceneData:
    scene_index: int
    start_offset: int
    end_offset: int
    location: str
    time: str
    characters: list[str] = field(default_factory=list)
    boundary_type: str = "scene_break"


def parse_chapter_file(path: Path) -> tuple[dict, str]:
    """Parse YAML frontmatter chapter file. Returns (metadata, content)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            content = parts[2].lstrip("\n")
            return metadata, content
    return {}, text


def build_agent(model_name: str, base_url: str, api_key: str) -> Agent:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    return Agent(model=model, output_type=SegmentationResult, system_prompt=SYSTEM_PROMPT)


def find_end_offset(chapter_text: str, end_text: str, search_from: int) -> int:
    """Find position right after end_text in chapter_text, searching from search_from.

    Returns the offset immediately after end_text (= start of next scene).
    Raises ValueError if end_text is not found.
    """
    pos = chapter_text.find(end_text, search_from)
    if pos == -1:
        raise ValueError(
            f"Could not locate end_text in chapter (searching from offset {search_from}).\n"
            f"  end_text = {end_text!r}\n"
            f"  context  = {chapter_text[search_from:search_from + 200]!r}"
        )
    return pos + len(end_text)


def build_user_prompt(summary: str, chunk: str, cursor: int, carry_info: str) -> str:
    parts = []
    if summary:
        parts.append(f"[Previously processed — summary]\n{summary}")
    if carry_info:
        parts.append(f"[Currently inside a scene]\n{carry_info}\nThe text window below is a continuation of this scene (or may transition to a new one).")
    parts.append(f"[Text window — cursor at character offset {cursor}]\n{chunk}")
    parts.append(
        "[Task]\n"
        "Identify all scenes in the text window above.\n"
        "For each complete scene output its metadata and end_text.\n"
        "For the last scene: if it extends beyond this window, set end_text='' and has_more=true."
    )
    return "\n\n".join(parts)


async def segment_chapter(chapter_text: str, agent: Agent) -> list[SceneData]:
    """Progressive LLM segmentation. Returns SceneData list with exact codepoint offsets."""
    total_length = len(chapter_text)
    all_scenes: list[SceneData] = []
    cursor = 0
    summary = ""
    carry_meta: SceneSegment | None = None

    while cursor < total_length:
        chunk = chapter_text[cursor:cursor + CHUNK_SIZE]
        carry_info = (
            f"location={carry_meta.location!r}, time={carry_meta.time!r}, "
            f"characters={carry_meta.characters}"
        ) if carry_meta else ""

        user_prompt = build_user_prompt(summary, chunk, cursor, carry_info)
        result = await agent.run(user_prompt)
        data: SegmentationResult = result.response  # type: ignore[assignment]

        if not data.scenes:
            # LLM returned nothing — force advance to avoid infinite loop
            cursor += CHUNK_SIZE // 2
            continue

        batch_cursor = cursor

        for i, seg in enumerate(data.scenes):
            scene_start = batch_cursor
            is_last = (i == len(data.scenes) - 1)

            if is_last and data.has_more:
                # Scene continues beyond this window; carry metadata to next batch
                carry_meta = seg
                break

            if is_last and not data.has_more:
                # Last scene of the chapter — always ends at total_length
                all_scenes.append(SceneData(
                    scene_index=len(all_scenes),
                    start_offset=scene_start,
                    end_offset=total_length,
                    location=seg.location,
                    time=seg.time,
                    characters=seg.characters,
                    boundary_type=seg.boundary_type,
                ))
                batch_cursor = total_length
                carry_meta = None
                break

            # Non-last scene — use end_text to find exact boundary
            if not seg.end_text:
                # LLM omitted end_text for a non-last scene; treat as carry
                carry_meta = seg
                break

            scene_end = find_end_offset(chapter_text, seg.end_text, batch_cursor)
            all_scenes.append(SceneData(
                scene_index=len(all_scenes),
                start_offset=scene_start,
                end_offset=scene_end,
                location=seg.location,
                time=seg.time,
                characters=seg.characters,
                boundary_type=seg.boundary_type,
            ))
            batch_cursor = scene_end
            carry_meta = None

        # Advance cursor past the last confirmed scene boundary
        if batch_cursor > cursor:
            cursor = batch_cursor
        else:
            # No progress in this batch — force advance to avoid infinite loop
            cursor += max(CHUNK_SIZE // 2, 1)

        summary = data.summary

        if not data.has_more:
            break

    return all_scenes


def verify_coverage(scenes: list[SceneData], total_length: int) -> None:
    """Assert scenes cover [0, total_length) with no gaps or overlaps."""
    if not scenes:
        raise ValueError("No scenes produced")
    if scenes[0].start_offset != 0:
        raise ValueError(f"First scene starts at {scenes[0].start_offset}, expected 0")
    for i in range(len(scenes) - 1):
        if scenes[i].end_offset != scenes[i + 1].start_offset:
            raise ValueError(
                f"Gap/overlap between scene {i} (end={scenes[i].end_offset}) "
                f"and scene {i + 1} (start={scenes[i + 1].start_offset})"
            )
    if scenes[-1].end_offset != total_length:
        raise ValueError(
            f"Last scene ends at {scenes[-1].end_offset}, expected {total_length}"
        )


def write_manifest(output_dir: Path, chapter_meta: dict, scenes: list[SceneData]) -> Path:
    manifest = {
        "chapter_index": chapter_meta.get("index", 0),
        "chapter_title": chapter_meta.get("title", ""),
        "total_length": scenes[-1].end_offset if scenes else 0,
        "scene_count": len(scenes),
        "scenes": [
            {
                "scene_index": s.scene_index,
                "location": s.location,
                "time": s.time,
                "characters": s.characters,
                "start_offset": s.start_offset,
                "end_offset": s.end_offset,
                "file": f"scene_{s.scene_index:03d}.txt",
            }
            for s in scenes
        ],
    }
    path = output_dir / "scenes_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_scene_files(output_dir: Path, chapter_text: str, scenes: list[SceneData]) -> None:
    for s in scenes:
        content = chapter_text[s.start_offset:s.end_offset]
        (output_dir / f"scene_{s.scene_index:03d}.txt").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment a novel chapter into scenes using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m extract.scene ch003.txt\n"
            "  uv run python -m extract.scene ch003.txt scenes/ch003/ --force\n"
            "  uv run python -m extract.scene ch003.txt --model openai/gpt-4o\n"
        ),
    )
    parser.add_argument("chapter_file", type=Path, help="YAML frontmatter chapter .txt file")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Output directory (default: <chapter_file>/../scenes/ch{index:03d}/)")
    parser.add_argument("--model",
                        default=os.environ.get("YORISHIRO_MODEL", "anthropic/claude-opus-4-6"),
                        help="Model name (env: YORISHIRO_MODEL, default: anthropic/claude-opus-4-6)")
    parser.add_argument("--base-url",
                        default=os.environ.get("YORISHIRO_BASE_URL", "https://openrouter.ai/api/v1"),
                        help="API base URL (env: YORISHIRO_BASE_URL, default: OpenRouter)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    args = parser.parse_args()

    chapter_file: Path = args.chapter_file
    if not chapter_file.exists():
        print(f"Error: {chapter_file} not found", file=sys.stderr)
        sys.exit(1)

    chapter_meta, chapter_text = parse_chapter_file(chapter_file)
    chapter_index = chapter_meta.get("index", 0)
    total_length = len(chapter_text)

    output_dir: Path = args.output_dir or (
        chapter_file.parent.parent / "scenes" / f"ch{chapter_index:03d}"
    )

    manifest_path = output_dir / "scenes_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Skipping: {manifest_path} already exists (use --force to overwrite)")
        sys.exit(0)

    api_key = os.environ.get("YORISHIRO_OPENAI_API_KEY")
    if not api_key:
        print("Error: YORISHIRO_OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args.model, args.base_url, api_key)

    print(f"Segmenting {chapter_file.name} ({total_length} chars) with {args.model} ...")
    try:
        scenes = asyncio.run(segment_chapter(chapter_text, agent))
    except ValueError as e:
        print(f"Error during segmentation: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying offsets ...")
    try:
        verify_coverage(scenes, total_length)
    except ValueError as e:
        print(f"Offset verification failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {len(scenes)} scenes to {output_dir} ...")
    write_scene_files(output_dir, chapter_text, scenes)
    write_manifest(output_dir, chapter_meta, scenes)

    print(f"Done.")
    for s in scenes:
        chars_label = f"[{s.start_offset}:{s.end_offset}]"
        print(f"  scene_{s.scene_index:03d}.txt  {chars_label:20s}  {s.location} / {s.time}")


if __name__ == "__main__":
    main()
