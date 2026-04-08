"""Scene segmentation engine for novel chapters."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import regex as _regex
import yaml
from pydantic import BaseModel, Field

from yorishiro.tasks.registry import StepRuntime
from yorishiro.utils import is_output_stale


INITIAL_CHUNK_SIZE = 8000
MAX_CHUNK_SIZE = 64000
MAX_RETRIES = 5
BOUNDARY_TYPES = [
    "chapter_start",
    "location_change",
    "time_jump",
    "pov_change",
    "narrative_break",
    "non_narrative",
]


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
    """One scene. Starts at cursor or where the previous scene ended."""

    location: str = Field(description="Scene location in source material language")
    time: str = Field(description="Time of day in source material language")
    characters: list[str] = Field(description="Character names exactly as they appear in this scene")
    boundary_type: str = Field(
        description="Type of boundary that ends this scene",
        json_schema_extra={"enum": BOUNDARY_TYPES},
    )
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


class _AgentRunResult(Protocol):
    output: SegmentationResult


class SegmentationAgent(Protocol):
    async def run(self, prompt: str) -> _AgentRunResult: ...


@dataclass
class SceneData:
    scene_index: int
    start_offset: int
    end_offset: int
    location: str
    time: str
    characters: list[str] = field(default_factory=list)
    boundary_type: str = "scene_break"


@dataclass(frozen=True)
class SceneSegmentationConfig:
    initial_chunk_size: int = INITIAL_CHUNK_SIZE
    max_chunk_size: int = MAX_CHUNK_SIZE

    @classmethod
    def from_step_config(cls, step_config: dict[str, Any] | None) -> SceneSegmentationConfig:
        cfg = step_config or {}
        initial_chunk_size = int(cfg.get("initial_chunk_size", INITIAL_CHUNK_SIZE))
        max_chunk_size = int(cfg.get("max_chunk_size", MAX_CHUNK_SIZE))
        if initial_chunk_size <= 0:
            raise ValueError(f"initial_chunk_size must be > 0, got {initial_chunk_size}")
        if max_chunk_size < initial_chunk_size:
            raise ValueError(
                f"max_chunk_size must be >= initial_chunk_size, got {max_chunk_size} < {initial_chunk_size}"
            )
        return cls(initial_chunk_size=initial_chunk_size, max_chunk_size=max_chunk_size)


@dataclass
class SegmentationState:
    total_length: int
    cursor: int = 0
    chunk_size: int = INITIAL_CHUNK_SIZE
    summary: str = ""
    failed_end_text: str = ""
    retry_count: int = 0
    last_scenes: list[SceneData] = field(default_factory=list)

    @classmethod
    def create(cls, total_length: int, config: SceneSegmentationConfig) -> SegmentationState:
        return cls(total_length=total_length, chunk_size=config.initial_chunk_size)

    def current_chunk(self, chapter_text: str) -> str:
        return chapter_text[self.cursor:self.cursor + self.chunk_size]

    def previous_chunk(self, chapter_text: str) -> str:
        return "\n".join(chapter_text[scene.start_offset:scene.end_offset] for scene in self.last_scenes)

    def reset_after_progress(self, next_cursor: int, config: SceneSegmentationConfig) -> None:
        self.cursor = next_cursor
        self.chunk_size = config.initial_chunk_size
        self.failed_end_text = ""
        self.retry_count = 0

    def grow_chunk_or_raise(self, config: SceneSegmentationConfig, message: str) -> None:
        self.retry_count += 1
        if self.retry_count >= MAX_RETRIES:
            raise RuntimeError(message)
        self.chunk_size = min(self.chunk_size * 2, config.max_chunk_size)


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
    chapter_text: str,
    end_text: str,
    search_from: int,
    search_limit: int | None = None,
) -> int:
    """Find position right after end_text in chapter_text, searching from search_from."""
    end_clean = "".join(end_text.split())
    end = (search_from + search_limit) if search_limit is not None else len(chapter_text)
    pairs = [(search_from + i, c) for i, c in enumerate(chapter_text[search_from:end]) if not c.isspace()]
    norm_str = "".join(c for _, c in pairs)

    pat = re.escape(end_clean)
    if "\\n" in end_clean:
        pat = pat + "|" + re.escape(end_clean.replace("\\n", ""))
    m = re.search(pat, norm_str)
    if m is not None:
        return pairs[m.end() - 1][0] + 1

    if len(end_clean) < 10:
        raise ValueError(
            f"Could not locate end_text in chapter (searching from offset {search_from}).\n"
            f"  end_text = {end_clean!r}\n"
            f"  context  = {chapter_text[search_from:search_from + 200]!r}"
        )

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
    chapter_end_note = (
        "This is the END of the chapter. The last scene must end here; set has_more=false."
        if is_final_chunk
        else "This is NOT the end of the chapter. More text follows after this window."
    )
    parts.append(
        f"[Task]\n"
        f"{chapter_end_note}\n"
        f"Identify all scenes in the novel_text above.\n"
        f"For each complete scene output its metadata and end_text. **DO NOT** mess up with end_text from previous scenes BEFORE cursor.\n"
        f"For the last scene: if it extends beyond this window, set end_text='' and has_more=true."
    )
    return "\n\n".join(parts)


def append_scene(
    scenes: list[SceneData],
    start_offset: int,
    end_offset: int,
    seg: SceneSegment,
) -> SceneData:
    if end_offset <= start_offset:
        raise ValueError(f"Empty/inverted scene: start={start_offset}, end={end_offset}")
    if scenes and scenes[-1].end_offset != start_offset:
        raise ValueError(
            f"Continuity gap: previous scene ends at {scenes[-1].end_offset}, "
            f"new scene starts at {start_offset}"
        )
    scene = SceneData(
        scene_index=len(scenes),
        start_offset=start_offset,
        end_offset=end_offset,
        location=seg.location,
        time=seg.time,
        characters=seg.characters,
        boundary_type=seg.boundary_type,
    )
    scenes.append(scene)
    return scene


def process_scene_batch(
    chapter_text: str,
    state: SegmentationState,
    data: SegmentationResult,
) -> tuple[int, bool]:
    batch_cursor = state.cursor
    reached_end = False
    state.last_scenes = []

    for i, seg in enumerate(data.scenes):
        scene_start = batch_cursor
        is_last = i == len(data.scenes) - 1

        if is_last and data.has_more:
            break

        if is_last and not data.has_more:
            append_scene(state.last_scenes, scene_start, state.total_length, seg)
            batch_cursor = state.total_length
            reached_end = True
            break

        if not seg.end_text:
            break

        scene_end = find_end_offset(
            chapter_text,
            seg.end_text,
            batch_cursor,
            search_limit=state.chunk_size + len(seg.end_text) + 500,
        )
        append_scene(state.last_scenes, scene_start, scene_end, seg)
        batch_cursor = scene_end

    return batch_cursor, reached_end


def grow_chunk_or_raise(
    state: SegmentationState,
    config: SceneSegmentationConfig,
    reason: str,
) -> None:
    state.grow_chunk_or_raise(
        config,
        f"{reason} at offset {state.cursor} after {MAX_RETRIES} retries",
    )


async def segment_chapter(
    chapter_text: str,
    agent: SegmentationAgent,
    segmentation_config: SceneSegmentationConfig | None = None,
) -> list[SceneData]:
    """Progressive LLM segmentation. Returns SceneData list with exact codepoint offsets."""
    config = segmentation_config or SceneSegmentationConfig()
    all_scenes: list[SceneData] = []
    state = SegmentationState.create(total_length=len(chapter_text), config=config)

    while state.cursor < state.total_length:
        chunk = state.current_chunk(chapter_text)
        is_final_chunk = state.cursor + len(chunk) >= state.total_length

        try:
            print(f"Calling model, current scene len = {len(all_scenes)}")
            result = await agent.run(
                build_user_prompt(
                    state.summary,
                    state.previous_chunk(chapter_text),
                    chunk,
                    state.cursor,
                    state.failed_end_text,
                    is_final_chunk,
                )
            )
            data = result.output
            print(f"  Result: scene len = {len(data.scenes)}, summary = {data.summary}")
        except Exception as e:
            print(f"  Warning: LLM error at offset {state.cursor}: {e}", file=sys.stderr)
            grow_chunk_or_raise(state, config, "LLM kept failing")
            continue

        state.summary = data.summary if data.summary != '""' else ""

        if not data.scenes:
            grow_chunk_or_raise(state, config, "LLM returned no scenes")
            continue

        try:
            batch_cursor, reached_end = process_scene_batch(chapter_text, state, data)
        except ValueError as e:
            print(f"  Warning: {e}. Continuing from last good cut.", file=sys.stderr)
            state.failed_end_text = data.scenes[-1].end_text
            grow_chunk_or_raise(state, config, "Stalled")
            continue

        if batch_cursor > state.cursor:
            all_scenes.extend(state.last_scenes)
            state.reset_after_progress(batch_cursor, config)
        else:
            grow_chunk_or_raise(state, config, "Stalled")

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
        raise ValueError(f"Last scene ends at {scenes[-1].end_offset}, expected {total_length}")


def write_manifest(output_dir: Path, chapter_meta: dict, scenes: list[SceneData], chapter_stem: str = "") -> Path:
    manifest = {
        "chapter_index": chapter_meta.get("index", 0),
        "chapter_stem": chapter_stem,
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


def process_chapter(
    chapter_file: Path,
    output_dir: Path,
    force: bool,
    runtime: StepRuntime,
    segmentation_config: SceneSegmentationConfig | None = None,
    material_yaml: Path | None = None,
) -> bool:
    """Process a single chapter file."""
    chapter_meta, chapter_text = parse_chapter_file(chapter_file)
    total_length = len(chapter_text)

    manifest_path = output_dir / "scenes_manifest.json"
    source_files = [chapter_file]
    if material_yaml is not None:
        source_files.append(material_yaml)
    if not force and not is_output_stale(manifest_path, source_files):
        print(f"Skipping {chapter_file.name}: output is up to date")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    agent = runtime.agent(
        output_type=SegmentationResult,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"Segmenting {chapter_file.name} ({total_length} chars) ...")

    scenes = asyncio.run(
        segment_chapter(
            chapter_text,
            cast(SegmentationAgent, agent),
            segmentation_config=segmentation_config,
        )
    )

    print("Verifying offsets ...")
    verify_coverage(scenes, total_length)

    print(f"Writing {len(scenes)} scenes to {output_dir} ...")
    write_scene_files(output_dir, chapter_text, scenes)
    write_manifest(output_dir, chapter_meta, scenes, chapter_stem=chapter_file.stem)

    print("Done.")
    for s in scenes:
        chars_label = f"[{s.start_offset}:{s.end_offset}]"
        print(f"  scene_{s.scene_index:03d}.txt  {chars_label:20s}  {s.location} / {s.time}")
    return True
