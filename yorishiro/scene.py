"""Automated LLM-based scene segmentation for novel chapters.

Usage:
    # Legacy mode:
    uv run python -m yorishiro.scene <chapter_file> [output_dir]
        [--model MODEL] [--base-url URL] [--api-key-env VAR] [--force]
        [--thinking {none,low,medium,high}]
    
    # Project mode:
    uv run python -m yorishiro.scene --project <project_dir> --source <source_id>
        [--chapter <index>] [--model MODEL] [--force]

Example:
    uv run python -m yorishiro.scene material/processed/novel/CPK/chapters/ch003.txt \\
                                    material/processed/novel/CPK/scenes/ch003
    uv run python -m yorishiro.scene --project projects/CPK --source cpk-novel --chapter 3

Input:
    YAML frontmatter chapter file produced by yorishiro.chapters_epub

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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import regex as _regex
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from yorishiro.agent_utils import add_model_args, build_agent_from_args
from yorishiro.project import ModelConfig, Project, find_project


INITIAL_CHUNK_SIZE = 8000   # chars per LLM batch
MAX_CHUNK_SIZE = 64000      # grows on stall; covers very long scenes


SYSTEM_PROMPT = """You are a professional narrative structure analyst. Your task is to segment a novel chapter into independent scenes.

## Source Material Language

ALL metadata (location, time, characters) must use the SAME language as the source material. Detect the language from the text.

## Summarize Previous Context First

Before identifying the scenes, summarize the story BEFORE the cursor. You can combine previous summary with last few scenes before the cursor.
If none of the text is available before the cursor, leave an empty string "" in the summary.
Try to keep summary under or around 4000 characters.

## Scene Definition

A scene is a narrative unit with:
- **Same location**: Events occur in the same physical/virtual space
- **Same time**: Continuous time period, no significant jumps
- **Same POV**: Primarily the same character's perspective
- **Coherent events**: Direct causal relationship between events

## Scene Boundary Signals (split ONLY on these)

1. Location change — character moves to a different place
2. Time jump — significant time progression (hours, days later, etc.)
3. POV change — perspective character changes
4. Narrative break — explicit separator (※, ──, ・・・ as scene break, etc.)
5. Significant context/state change — atmosphere shifting from relaxed to tense, a game just started, etc.

Also judge on whether a scene cut would result in reasonable size — It's probably not reasonable to generate a very short scene.

**IMPORTANT**: Carefully think which scene each sentence belongs to. Pay attention to phrases indicating changing of time or location.
Identify both ending text of current scene and starting text of next scene, though only ending text is needed for the response.

## Cut Point Requirements

**CRITICAL**: Cuts must be at natural narrative boundaries, NEVER mid-sentence:
- After narrative separators (※ or ──)
- At paragraph/section breaks
- At sentence endings (。 or ！or ？or similar)
- Before a clear new beginning (indication of new location/time/POV)

Think about 3-4 potential cut point for each scene, and evaluate which one is the best by the rules defined in this document.

**DO NOT** cut in the middle of a flowing sentence or dialogue.

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

## Quality Checklist Before Generating Response

- [ ] All metadata uses source material language
- [ ] Summary is properly generated
- [ ] Each split has clear boundary justification
- [ ] No single scene that is too short or too long (usually indication of non-optimal scene cut)
- [ ] No text omission or duplication
"""


class SceneSegment(BaseModel):
    """One scene. Starts at cursor (or where the previous scene ended)."""

    location: str = Field(description="Scene location in source material language")
    time: str = Field(description="Time of day in source material language")
    characters: list[str] = Field(description="Character names exactly as they appear in this scene")
    boundary_type: str = Field(description="Type of boundary that ends this scene", json_schema_extra={"enum": [
        "chapter_start",
        "location_change",
        "time_jump",
        "pov_change",
        "narrative_break",
        "non_narrative",
    ]})
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
        description=(
            "Summary of the story so far BEFORE the cursor. "
            "Can be a combine of previous summary with last few scenes text. "
            "Empty string '' if there is no previous story summary and scene texts. "
            "Used as a compacted context."
        )
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



def find_end_offset(
    chapter_text: str, end_text: str, search_from: int,
    search_limit: int | None = None,
) -> int:
    """Find position right after end_text in chapter_text, searching from search_from.

    Both end_text and the search region are whitespace-stripped for matching;
    the returned offset is a codepoint position in the original chapter_text.
    For end_text with >= 10 non-whitespace chars, falls back to fuzzy matching
    allowing up to floor(len/10) insert/delete/replace edits; the last two chars
    must always match exactly.
    Returns the offset immediately after end_text (= start of next scene).
    Raises ValueError if end_text is not found.

    search_limit: if given, cap the search to this many original characters after
    search_from.  Keeps fuzzy matching fast by avoiding scanning the whole chapter.
    """
    end_clean = "".join(end_text.split())
    end = (search_from + search_limit) if search_limit is not None else len(chapter_text)
    pairs = [(search_from + i, c) for i, c in enumerate(chapter_text[search_from:end]) if not c.isspace()]
    norm_str = "".join(c for _, c in pairs)

    # Exact match on whitespace-stripped text
    pat = re.escape(end_clean)
    # Special fix for Gemini models - it escapes newline in the string
    if "\\n" in end_clean:
        pat = pat + "|" + end_clean.replace("\\n", "")
    m = re.search(pat, norm_str)
    if m is not None:
        return pairs[m.end() - 1][0] + 1

    if len(end_clean) < 10:
        raise ValueError(
            f"Could not locate end_text in chapter (searching from offset {search_from}).\n"
            f"  end_text = {end_clean!r}\n"
            f"  context  = {chapter_text[search_from:search_from + 200]!r}"
        )

    # Fuzzy fallback: match body fuzzily, tail exactly
    max_errors = len(end_clean) // 10
    body, tail = end_clean[:-2], end_clean[-2:]
    pat = _regex.compile(rf"(?:{_regex.escape(body)}){{e<={max_errors}}}{_regex.escape(tail)}")
    fm = pat.search(norm_str)
    if fm is not None:
        return pairs[fm.end() - 1][0] + 1

    raise ValueError(
        f"Could not locate end_text in chapter (searching from offset {search_from}).\n"
        f"  end_text = {end_clean!r}\n"
        f"  context  = {chapter_text[search_from:search_from + 200]!r}"
    )


def build_user_prompt(
    summary: str,
    previous_chunk: str,
    chunk: str,
    cursor: int,
    failed_end_text: str = "",
    is_final_chunk: bool = False,
) -> str:
    parts = []
    if not summary and not previous_chunk:
        parts.append("[This is at START of the chapter — there is NO text before the cursor]")
    if summary:
        parts.append(f"[Previously processed — summary of the story so far]\n{summary}")
    if previous_chunk:
        parts.append(f"[Previously processed — last few scenes BEFORE cursor]\n{previous_chunk}")
    if failed_end_text:
        parts.append(
            f"[Previous attempt failed]\n"
            f"The end_text {failed_end_text!r} could not be located in the source text. "
            f"It was likely hallucinated. Please choose a different end_text that is verbatim from the text inside <novel_text>."
        )
    parts.append(
        f"[Text window — cursor at character offset {cursor}]\n"
        f"<novel_text>\n{chunk}\n</novel_text>"
    )
    if is_final_chunk:
        chapter_end_note = "This is the END of the chapter. The last scene must end here; set has_more=false."
    else:
        chapter_end_note = "This is NOT the end of the chapter. More text follows after this window."
    parts.append(
        f"[Task]\n"
        f"{chapter_end_note}\n"
        f"Identify all scenes in the novel_text above.\n"
        f"For each complete scene output its metadata and end_text. **DO NOT** mess up with end_text from previous scenes BEFORE cursor.\n"
        f"For the last scene: if it extends beyond this window, set end_text='' and has_more=true."
    )
    return "\n\n".join(parts)


def _append_scene(
    scenes: list[SceneData],
    start_offset: int,
    end_offset: int,
    seg: SceneSegment | None,
) -> None:
    """Append a scene with in-loop continuity validation. Raises ValueError on violation."""
    if end_offset <= start_offset:
        raise ValueError(f"Empty/inverted scene: start={start_offset}, end={end_offset}")
    if scenes and scenes[-1].end_offset != start_offset:
        raise ValueError(
            f"Continuity gap: previous scene ends at {scenes[-1].end_offset}, "
            f"new scene starts at {start_offset}"
        )
    scenes.append(SceneData(
        scene_index=len(scenes),
        start_offset=start_offset,
        end_offset=end_offset,
        location=seg.location if seg else "N/A",
        time=seg.time if seg else "N/A",
        characters=seg.characters if seg else [],
        boundary_type=seg.boundary_type if seg else "chapter_start",
    ))


async def segment_chapter(chapter_text: str, agent: Agent[None, SegmentationResult]) -> list[SceneData]:
    """Progressive LLM segmentation. Returns SceneData list with exact codepoint offsets.

    - cursor never advances unless a scene boundary is confirmed
    - chunk_size grows when no progress is made, keeping the full current scene in context
    - LLM/matching errors resume from the last confirmed scene cut
    - continuity is validated after each scene is appended
    """
    MAX_RETRIES = 5

    total_length = len(chapter_text)
    all_scenes: list[SceneData] = []
    last_scenes: list[SceneData] = []
    cursor = 0          # always = start of current unfinished scene
    chunk_size = INITIAL_CHUNK_SIZE
    summary = ""
    failed_end_text = ""  # end_text that failed matching last attempt
    retry_count = 0     # resets on any forward progress

    while cursor < total_length:
        chunk = chapter_text[cursor:cursor + chunk_size]
        is_final_chunk = (cursor + len(chunk) >= total_length)
        previous_chunk = "\n".join(chapter_text[scene.start_offset:scene.end_offset] for scene in last_scenes)
        last_scenes = []

        # --- LLM call ---
        try:
            print(f"Calling model, current scene len = {len(all_scenes)}")
            result = await agent.run(
                build_user_prompt(summary, previous_chunk, chunk, cursor, failed_end_text, is_final_chunk)
            )
            data = result.output
            print(f"  Result: scene len = {len(data.scenes)}, summary = {data.summary}")
        except Exception as e:
            print(f"  Warning: LLM error at offset {cursor}: {e}", file=sys.stderr)
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise RuntimeError(f"LLM kept failing at offset {cursor} after {MAX_RETRIES} retries")
            chunk_size = min(chunk_size * 2, MAX_CHUNK_SIZE)
            continue

        summary = data.summary if data.summary != '""' else ""

        # --- No scenes returned ---
        if not data.scenes:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise RuntimeError(f"LLM returned no scenes at offset {cursor} after {MAX_RETRIES} retries")
            chunk_size = min(chunk_size * 2, MAX_CHUNK_SIZE)
            continue

        # --- Process scenes in batch ---
        batch_cursor = cursor
        reached_end = False

        for i, seg in enumerate(data.scenes):
            scene_start = batch_cursor
            is_last = (i == len(data.scenes) - 1)

            if is_last and data.has_more:
                # Scene continues beyond window — carry to next batch
                break

            if is_last and not data.has_more:
                # Final scene of the chapter
                try:
                    _append_scene(all_scenes, scene_start, total_length, seg)
                except ValueError as e:
                    print(f"  Warning: {e}. Stopping at last confirmed cut.", file=sys.stderr)
                    break
                batch_cursor = total_length
                reached_end = True
                break

            # Non-last scene: locate boundary via end_text
            if not seg.end_text:
                break

            try:
                scene_end = find_end_offset(
                    chapter_text, seg.end_text, batch_cursor,
                    search_limit=chunk_size + len(seg.end_text) + 500,
                )
                _append_scene(all_scenes, scene_start, scene_end, seg)
                _append_scene(last_scenes, scene_start, scene_end, seg)
            except ValueError as e:
                print(f"  Warning: {e}. Continuing from last good cut.", file=sys.stderr)
                failed_end_text = seg.end_text
                break

            batch_cursor = scene_end

        # --- Advance cursor or grow chunk ---
        if batch_cursor > cursor:
            cursor = batch_cursor
            chunk_size = INITIAL_CHUNK_SIZE  # reset after progress
            failed_end_text = ""
            retry_count = 0
        else:
            # No progress — keep cursor, grow chunk so full scene stays in context
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise RuntimeError(f"Stalled at offset {cursor} after {MAX_RETRIES} retries with no progress")
            chunk_size = min(chunk_size * 2, MAX_CHUNK_SIZE)

        if reached_end:
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
            "  # Legacy mode:\n"
            "  uv run python -m yorishiro.scene ch003.txt\n"
            "  uv run python -m yorishiro.scene ch003.txt scenes/ch003/ --force\n"
            "  uv run python -m yorishiro.scene ch003.txt --model openai/gpt-4o\n"
            "\n"
            "  # Project mode:\n"
            "  uv run python -m yorishiro.scene --project projects/CPK --source cpk-novel --chapter 3\n"
            "  uv run python -m yorishiro.scene --project projects/CPK --source cpk-novel --all\n"
        ),
    )
    
    # Project mode arguments
    parser.add_argument("--project", type=Path, default=None,
                        help="Project directory (enables project mode)")
    parser.add_argument("--source", type=str, default=None,
                        help="Source ID within project (required with --project)")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Chapter index to process (0-based, use with --project)")
    parser.add_argument("--all", action="store_true",
                        help="Process all chapters (use with --project)")
    
    # Legacy mode arguments
    parser.add_argument("chapter_file", type=Path, nargs="?", default=None,
                        help="YAML frontmatter chapter .txt file (legacy mode)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None,
                        help="Output directory (legacy mode)")
    
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    add_model_args(parser)
    args = parser.parse_args()

    # Determine mode and resolve paths
    project: Project | None = None
    source_id: str | None = None
    chapter_file: Path | None = None
    output_dir: Path | None = None
    config: ModelConfig | None = None

    if args.project:
        # Project mode
        project = Project.load(args.project)
        
        if not args.source:
            print("Error: --source is required when using --project", file=sys.stderr)
            sys.exit(1)
        source_id = args.source
        
        source_config = project.get_source(source_id)
        if not source_config:
            print(f"Error: Source '{source_id}' not found in project", file=sys.stderr)
            sys.exit(1)
        
        config = project.resolved_model_config("scene")
        
        chapters_dir = project.source_dir(source_id) / "chapters"
        
        if args.all:
            # Process all chapters
            chapter_indices = project.list_chapters(source_id)
            if not chapter_indices:
                print(f"Error: No chapters found in {chapters_dir}", file=sys.stderr)
                sys.exit(1)
            
            for idx in chapter_indices:
                process_chapter(
                    chapters_dir / f"ch{idx:03d}.txt",
                    project.source_dir(source_id) / "scenes" / f"ch{idx:03d}",
                    args.force,
                    args,
                    config,
                )
            return
        
        if args.chapter is not None:
            # Process specific chapter
            chapter_file = chapters_dir / f"ch{args.chapter:03d}.txt"
            output_dir = project.source_dir(source_id) / "scenes" / f"ch{args.chapter:03d}"
        else:
            # Find first unprocessed chapter
            chapter_indices = project.list_chapters(source_id)
            for idx in chapter_indices:
                manifest_path = project.source_dir(source_id) / "scenes" / f"ch{idx:03d}" / "scenes_manifest.json"
                if not manifest_path.exists():
                    chapter_file = chapters_dir / f"ch{idx:03d}.txt"
                    output_dir = project.source_dir(source_id) / "scenes" / f"ch{idx:03d}"
                    break
            if not chapter_file:
                print("All chapters already processed", file=sys.stderr)
                sys.exit(0)
    
    elif args.chapter_file:
        # Legacy mode
        chapter_file = args.chapter_file
        if not chapter_file.exists():
            print(f"Error: {chapter_file} not found", file=sys.stderr)
            sys.exit(1)
        
        chapter_meta, _ = parse_chapter_file(chapter_file)
        chapter_index = chapter_meta.get("index", 0)
        output_dir = args.output_dir or (
            chapter_file.parent.parent / "scenes" / f"ch{chapter_index:03d}"
        )
    
    else:
        # Try auto-detection
        project = find_project(Path.cwd())
        if project and project.sources:
            source_id = project.sources[0].id
            config = project.resolved_model_config("scene")
            chapter_indices = project.list_chapters(source_id)
            if chapter_indices:
                chapter_file = project.source_dir(source_id) / "chapters" / f"ch{chapter_indices[0]:03d}.txt"
                output_dir = project.source_dir(source_id) / "scenes" / f"ch{chapter_indices[0]:03d}"
                print(f"Auto-detected project: {project.name}")
                print(f"Processing chapter {chapter_indices[0]}")
            else:
                print("Error: No chapters found", file=sys.stderr)
                sys.exit(1)
        else:
            parser.error("Either --project/--source or chapter_file is required")

    assert chapter_file is not None
    assert output_dir is not None
    process_chapter(
        chapter_file,
        output_dir,
        args.force,
        args,
        config,
    )


def process_chapter(
    chapter_file: Path,
    output_dir: Path,
    force: bool,
    args: argparse.Namespace,
    config: ModelConfig | None,
) -> None:
    """Process a single chapter file."""
    if not chapter_file.exists():
        print(f"Error: {chapter_file} not found", file=sys.stderr)
        sys.exit(1)

    chapter_meta, chapter_text = parse_chapter_file(chapter_file)
    total_length = len(chapter_text)

    manifest_path = output_dir / "scenes_manifest.json"
    if manifest_path.exists() and not force:
        print(f"Skipping: {manifest_path} already exists (use --force to overwrite)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        agent = build_agent_from_args(
            args,
            output_type=SegmentationResult,
            system_prompt=SYSTEM_PROMPT,
            config=config,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    model_display = args.model or (config.name if config else "unknown")
    print(f"Segmenting {chapter_file.name} ({total_length} chars) with {model_display} ...")
    
    try:
        scenes = asyncio.run(segment_chapter(chapter_text, agent))
    except ValueError as e:
        print(f"Error during segmentation: {e}", file=sys.stderr)
        sys.exit(1)

    print("Verifying offsets ...")
    try:
        verify_coverage(scenes, total_length)
    except ValueError as e:
        print(f"Offset verification failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {len(scenes)} scenes to {output_dir} ...")
    write_scene_files(output_dir, chapter_text, scenes)
    write_manifest(output_dir, chapter_meta, scenes)

    print("Done.")
    for s in scenes:
        chars_label = f"[{s.start_offset}:{s.end_offset}]"
        print(f"  scene_{s.scene_index:03d}.txt  {chars_label:20s}  {s.location} / {s.time}")


if __name__ == "__main__":
    main()
