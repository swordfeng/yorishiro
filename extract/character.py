"""Per-scene character extraction pipeline: generate character notes and finalize soul docs.

Usage:
    uv run python -m extract.character <scenes_base_dir>
        [--aliases-file PATH] [--output-dir PATH] [--souls-dir PATH]
        [--characters NAME [NAME ...]]
        [--batch-tokens N] [--cjk-ratio F]
        [--model MODEL] [--provider PROVIDER] [--api-key-env VAR]
        [--base-url URL] [--thinking {none,low,medium,high}]
        [--output-mode {tool,native,prompted}]
        [--no-finalize]

Example:
    uv run python -m extract.character material/processed/novel/CPK/scenes \\
        --characters 酒寄彩葉 かぐや --model anthropic/claude-opus-4-6

Input:
    <scenes_base_dir>/ -- scene text files and manifests (produced by extract.scene)
    character_aliases.json -- canonical-name → alias occurrence mapping
    souls/*.md -- seed soul docs (produced by extract.resolve_aliases)

Output:
    <output_dir>/{canonical_name}/ch{N:03d}.json -- per-chapter CharacterSceneNote arrays
    <output_dir>/{canonical_name}/insights.md -- accumulated soul doc insights (grows per batch)
    <souls_dir>/{canonical_name}.md -- finalized SOUL.md (overwritten after finalization pass)

Processing:
    1. Load all scenes with canonical character annotations from character_aliases.json.
    2. Load seed soul docs + accumulated insights per target character.
    3. Process scenes in batches (~batch_tokens tokens each).
       Each batch extracts CharacterSceneNote for each target character × scene,
       plus synthesizes SoulDocAppend insights per character (cross-scene observations
       not captured in per-scene notes and not already in the soul doc / insights).
    4. After all batches: finalization pass — produce full structured SOUL.md per character.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from extract.agent_utils import add_model_args, build_agent, resolve_api_key


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Relationship(BaseModel):
    target: str
    dynamic: str
    shift: str | None = None


class KnowledgeScope(BaseModel):
    facts_revealed: list[str]
    facts_hidden: list[str]


class CharacterSceneNote(BaseModel):
    """Structured extraction of one character's information in one scene."""

    character: str
    chapter_index: int
    scene_index: int
    source: str = "novel"
    active_persona: str
    dialogue_samples: list[str]
    language_traits: str
    emotional_state: str
    inferred_motivation: str
    internal_conflict: str
    relationships: list[Relationship]
    actions_taken: str
    actions_avoided: str
    decision_logic: str
    arc_marker: str
    knowledge_scope: KnowledgeScope


class SoulDocAppend(BaseModel):
    """Per-character synthesized insights from a batch, for soul doc accumulation."""

    canonical_name: str = Field(
        description="Canonical name of the character (must match a target character)."
    )
    negative_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Absolute behavioral limits observed across this batch's scenes, "
            "each with the underlying reason or value that drives the constraint."
        ),
    )
    behavioral_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Cross-scene consistent behavioral patterns: how the character acts under "
            "pressure, with intimacy, facing failure, in moral dilemmas, habitual rituals."
        ),
    )
    simulation_directives: list[str] = Field(
        default_factory=list,
        description=(
            "Meta-instructions for roleplay simulation: persona defaults, prohibited "
            "behaviors, persona-switching rules, fallback strategies."
        ),
    )
    relationship_insights: list[str] = Field(
        default_factory=list,
        description=(
            "Cross-scene relationship history, power dynamics, emotional weight, "
            "unresolved tensions not captured in per-scene relationship fields."
        ),
    )
    personality_synthesis: list[str] = Field(
        default_factory=list,
        description=(
            "Core values, deepest fears, defense mechanisms, cognitive patterns "
            "synthesized from observing this character across multiple scenes."
        ),
    )
    arc_notes: str = Field(
        default="",
        description="Character arc development and turning points observed in this batch.",
    )


class BatchExtractionResult(BaseModel):
    """LLM output for one scene batch."""

    notes: list[CharacterSceneNote] = Field(
        description=(
            "One CharacterSceneNote per (target character × scene) where the character "
            "actually appears. Identified by character + chapter_index + scene_index. "
            "Omit entirely for characters not present in a scene."
        )
    )
    soul_doc_appends: list[SoulDocAppend] = Field(
        default_factory=list,
        description=(
            "One SoulDocAppend per target character with genuinely new insights from "
            "this batch. Only include characters with something new to add. "
            "Do NOT repeat content already in the character's soul doc or insights."
        ),
    )


class SoulDocOutput(BaseModel):
    """Finalization pass output: a complete structured SOUL.md."""

    content: str = Field(
        description="Complete SOUL.md document in markdown format.",
    )


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class CharacterInScene:
    alias: str      # exact alias as in scene manifest
    canonical: str  # resolved canonical name (equals alias if unresolved)


@dataclass
class SceneRecord:
    chapter: str
    chapter_index: int
    scene_index: int
    location: str
    time: str
    characters: list[CharacterInScene]         # all characters with canonical annotation
    target_characters: list[CharacterInScene]  # only the target ones
    text: str


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert literary analyst extracting structured character data from novel scenes.

## Your Task

You receive:
1. **Character Backgrounds**: Soul docs and accumulated insights for each target character \
appearing in this batch.
2. **Scene Batch**: Multiple scenes with metadata. Character names are annotated with their \
canonical names (「alias」→ canonical_name).

For each batch, produce two types of output:

### A. CharacterSceneNote (per character × scene)

For every target character that actually appears in a scene, extract one note. \
Characters not present in a scene must be omitted.

- `character`: Use the canonical name (the part after →, not the alias).
- `active_persona`: Which persona/identity is active (critical for composite characters).
- `dialogue_samples`: Exact verbatim original dialogue — never paraphrase or translate.
- `language_traits`: Observed speech patterns, register, catchphrases, humor style.
- `emotional_state`: Current emotional state in this scene.
- `inferred_motivation`: What is driving their behavior.
- `internal_conflict`: Inner conflict if present; empty string if none.
- `relationships`: Per-other-character dynamics and shifts observed in this scene.
- `actions_taken`: What they did.
- `actions_avoided`: What they could have done but didn't — negative constraint signals.
- `decision_logic`: Reasoning behind their choices.
- `arc_marker`: Whether this is a character development node; how it changes them.
- `knowledge_scope.facts_revealed`: What they learned in this scene.
- `knowledge_scope.facts_hidden`: What the reader knows that they don't.

### B. SoulDocAppend (per character, synthesized across the batch)

For each target character appearing in this batch, synthesize cross-scene insights that:
1. Are relevant to a final SOUL.md document (negative constraints, behavioral patterns, \
simulation directives, relationship history, personality synthesis, arc development)
2. Are **not** already captured by the per-scene CharacterSceneNote fields above
3. Are **not** already present in the character's soul doc or accumulated insights

Omit a character from soul_doc_appends if there is genuinely nothing new to add.

## Critical Rules

- Use **canonical names** in the `character` field — never aliases.
- `dialogue_samples` must be exact original text, never paraphrased or translated.
- `actions_avoided` requires special care: identify opportunities the character passed up.
- `active_persona` matters for characters with multiple identities or timelines.
- Do NOT translate or romanize names — preserve the original writing system.
- **Write all content fields in the source language** (Japanese, Chinese, etc.) — \
`emotional_state`, `inferred_motivation`, `internal_conflict`, `actions_taken`, \
`actions_avoided`, `decision_logic`, `arc_marker`, `language_traits`, `active_persona`, \
`knowledge_scope` fields, and all `SoulDocAppend` list items. \
Do NOT translate content into English. Structural labels stay in English.
"""

FINALIZATION_SYSTEM_PROMPT = """\
You are synthesizing a complete, production-ready character soul document (SOUL.md) for use \
in AI roleplay simulation.

## Input

You will receive for one character:
1. A draft seed soul doc
2. Accumulated cross-scene insights
3. All per-scene extraction notes (CharacterSceneNote records as JSON)

## Output Format

Produce a complete SOUL.md in markdown following this structure exactly. \
Do not invent facts not evidenced in the provided material; flag uncertain areas with a note.

---

# {Character Name}

> One-sentence identity summary

## 1. Core Identity
Core driving motivation; current life stage/role.

## 2. Personality Model

### 2.1 Core Values
### 2.2 Motivations & Desires
- Surface desire:
- Deep desire:
- Ultimate desire:
### 2.3 Core Fears & Avoidances
### 2.4 Character Traits
- Public persona:
- True self:
- Internal contradictions:
### 2.5 Cognitive Patterns

## 3. Voice & Language

### 3.1 Overall Register
### 3.2 Catchphrases & Signature Expressions
### 3.3 Sentence Style Preferences
### 3.4 Humor Style
### 3.5 Language Under Emotional Intensity
### 3.6 Dialogue Samples
Include original-language verbatim dialogue with scene context.

## 4. Relationships
For each significant relationship: nature, interaction patterns, key turning points.

## 5. Behavioral Patterns

### 5.1 Under Pressure / Conflict
### 5.2 With Intimacy / Trust
### 5.3 Facing Failure / Setbacks
### 5.4 Moral Dilemmas
### 5.5 Habitual Behaviors & Rituals

## 6. Negative Constraints
Explicit rules for what this character would NEVER do, each with the underlying reason. \
Critical for preventing out-of-character behavior in simulation.

## 7. Character Arc

### 7.1 Starting State
### 7.2 Key Turning Points
### 7.3 Ending State
### 7.4 Stage-by-Stage Personality Differences

## 8. World Knowledge

### Known Facts
### Unknown to Character
### Mistaken Beliefs

## 9. Simulation Directives
Direct instructions for the AI agent: default timeline to simulate, dialogue style rules, \
prohibited behaviors checklist, persona-switching triggers, fallback strategies.

---

Note: This document may be further refined in cross-source alignment passes.

Write the soul document in the same language as the source material (Japanese, Chinese, etc.). \
Section headings may stay in English for tooling compatibility, but all content should be in the source language.
"""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def agent_run_with_retry(agent, prompt: str, max_attempts: int = 3):
    """Run agent, retrying on output validation errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await agent.run(prompt)
        except (ValidationError, UnexpectedModelBehavior) as exc:
            last_exc = exc
            print(
                f"  [Output validation error attempt {attempt}/{max_attempts}] {exc} — retrying ...",
                file=sys.stderr,
            )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Token estimation / batching
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, cjk_ratio: float) -> int:
    return int(len(text) * cjk_ratio)


def build_batches(
    scenes: list[SceneRecord],
    batch_tokens: int,
    cjk_ratio: float = 0.65,
) -> list[list[SceneRecord]]:
    """Group scenes into batches not exceeding batch_tokens estimated tokens."""
    batches: list[list[SceneRecord]] = []
    current_batch: list[SceneRecord] = []
    current_tokens = 0
    for scene in scenes:
        scene_tokens = estimate_tokens(scene.text, cjk_ratio)
        if current_batch and current_tokens + scene_tokens > batch_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(scene)
        current_tokens += scene_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


# ---------------------------------------------------------------------------
# Alias / scene loading
# ---------------------------------------------------------------------------


def load_aliases(aliases_path: Path) -> dict[str, list[dict]]:
    """Load character_aliases.json, excluding UNRESOLVED."""
    data = json.loads(aliases_path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "UNRESOLVED"}


def build_alias_canonical_lookup(
    aliases: dict[str, list[dict]],
) -> dict[tuple[str, int, str], str]:
    """Build (chapter, scene_index, alias) → canonical_name lookup."""
    lookup: dict[tuple[str, int, str], str] = {}
    for canonical, occurrences in aliases.items():
        for occ in occurrences:
            key = (occ["chapter"], occ["scene"], occ["alias"])
            lookup[key] = canonical
    return lookup


def load_all_scenes(
    scenes_base_dir: Path,
    aliases: dict[str, list[dict]],
    target_characters: list[str],
) -> list[SceneRecord]:
    """Load all scenes annotated with canonical character names."""
    alias_lookup = build_alias_canonical_lookup(aliases)
    target_set = set(target_characters)
    scenes: list[SceneRecord] = []

    for manifest_path in sorted(scenes_base_dir.glob("*/scenes_manifest.json")):
        chapter = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chapter_index = manifest.get("chapter_index", 0)

        for scene in manifest.get("scenes", []):
            scene_index = scene["scene_index"]
            scene_file = manifest_path.parent / f"scene_{scene_index:03d}.txt"
            if not scene_file.exists():
                print(f"  Warning: {scene_file} not found, skipping", file=sys.stderr)
                continue

            characters: list[CharacterInScene] = []
            for alias in scene.get("characters", []):
                canonical: str = alias_lookup.get((chapter, scene_index, alias), alias) or alias
                characters.append(CharacterInScene(alias=alias, canonical=canonical))

            target_in_scene = [c for c in characters if c.canonical in target_set]

            scenes.append(SceneRecord(
                chapter=chapter,
                chapter_index=chapter_index,
                scene_index=scene_index,
                location=scene.get("location", ""),
                time=scene.get("time", ""),
                characters=characters,
                target_characters=target_in_scene,
                text=scene_file.read_text(encoding="utf-8"),
            ))

    return sorted(scenes, key=lambda s: (s.chapter, s.scene_index))


# ---------------------------------------------------------------------------
# Soul doc / insights loading
# ---------------------------------------------------------------------------


def load_soul_context(canonical_name: str, souls_dir: Path, output_dir: Path) -> str:
    """Return combined soul doc + accumulated insights text for a character."""
    parts: list[str] = []

    soul_path = souls_dir / f"{canonical_name}.md"
    if soul_path.exists():
        parts.append(f"### Soul Doc\n\n{soul_path.read_text(encoding='utf-8')}")

    insights_path = output_dir / canonical_name / "insights.md"
    if insights_path.exists():
        parts.append(
            f"### Accumulated Insights\n\n{insights_path.read_text(encoding='utf-8')}"
        )

    return "\n\n".join(parts) if parts else "(no background available yet)"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_batch_prompt(
    batch: list[SceneRecord],
    soul_contexts: dict[str, str],
    batch_num: int,
    total_batches: int,
) -> str:
    parts: list[str] = [f"# Batch {batch_num}/{total_batches}  ({len(batch)} scenes)\n"]

    if soul_contexts:
        parts.append("## Character Backgrounds\n")
        for name, context in soul_contexts.items():
            parts.append(f"### {name}\n\n{context}\n")

    parts.append("## Scene Batch\n")

    for scene in batch:
        annotations: list[str] = []
        for c in scene.characters:
            if c.alias != c.canonical:
                annotations.append(f"「{c.alias}」→ {c.canonical}")
            else:
                annotations.append(f"「{c.alias}」")
        char_str = ",  ".join(annotations) if annotations else "(none)"

        parts.append(
            f"---\n"
            f"**{scene.chapter} / scene_{scene.scene_index:03d}**"
            f"  | Location: {scene.location or '?'}"
            f"  | Time: {scene.time or '?'}"
            f"  | Characters: {char_str}\n\n"
            f"{scene.text}\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_chapter_notes(
    output_dir: Path,
    canonical_name: str,
    chapter_index: int,
    notes: list[dict],
) -> None:
    out_path = output_dir / canonical_name / f"ch{chapter_index:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notes_sorted = sorted(notes, key=lambda n: n.get("scene_index", 0))
    out_path.write_text(
        json.dumps(notes_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_insights(
    output_dir: Path,
    append: SoulDocAppend,
    batch_num: int,
    total: int,
) -> None:
    """Append batch insights to characters/{name}/insights.md."""
    insights_path = output_dir / append.canonical_name / "insights.md"
    insights_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"\n## Batch {batch_num}/{total}\n"]

    if append.negative_constraints:
        lines.append("### Negative Constraints\n")
        lines.extend(f"- {c}" for c in append.negative_constraints)
        lines.append("")
    if append.behavioral_patterns:
        lines.append("### Behavioral Patterns\n")
        lines.extend(f"- {p}" for p in append.behavioral_patterns)
        lines.append("")
    if append.simulation_directives:
        lines.append("### Simulation Directives\n")
        lines.extend(f"- {d}" for d in append.simulation_directives)
        lines.append("")
    if append.relationship_insights:
        lines.append("### Relationship Insights\n")
        lines.extend(f"- {r}" for r in append.relationship_insights)
        lines.append("")
    if append.personality_synthesis:
        lines.append("### Personality Synthesis\n")
        lines.extend(f"- {p}" for p in append.personality_synthesis)
        lines.append("")
    if append.arc_notes:
        lines.append("### Arc Notes\n")
        lines.append(append.arc_notes)
        lines.append("")

    with insights_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


async def process_all_batches(
    batches: list[list[SceneRecord]],
    target_characters: list[str],
    souls_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    api_key: str,
) -> None:
    total = len(batches)
    # Accumulate notes keyed by (canonical_name, chapter_index)
    all_notes: dict[tuple[str, int], list[dict]] = {}

    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{total}  ({len(batch)} scenes) ...", file=sys.stderr)

        # Only load soul contexts for target characters appearing in this batch
        chars_in_batch = {c.canonical for scene in batch for c in scene.target_characters}
        soul_contexts = {
            name: load_soul_context(name, souls_dir, output_dir)
            for name in target_characters
            if name in chars_in_batch
        }

        prompt = format_batch_prompt(batch, soul_contexts, i, total)

        agent = build_agent(
            model_name=args.model,
            provider_name=args.provider,
            api_key=api_key,
            base_url=args.base_url,
            output_type=BatchExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            thinking=args.thinking,
            output_mode=args.output_mode,
        )

        result = await agent_run_with_retry(agent, prompt)
        extraction: BatchExtractionResult = result.output

        for note in extraction.notes:
            key = (note.character, note.chapter_index)
            all_notes.setdefault(key, []).append(note.model_dump())

        for app in extraction.soul_doc_appends:
            if any([
                app.negative_constraints,
                app.behavioral_patterns,
                app.simulation_directives,
                app.relationship_insights,
                app.personality_synthesis,
                app.arc_notes,
            ]):
                append_insights(output_dir, app, i, total)

        print(
            f"  → {len(extraction.notes)} notes, "
            f"{len(extraction.soul_doc_appends)} soul doc appends",
            file=sys.stderr,
        )

    print("\nWriting character notes ...", file=sys.stderr)
    for (canonical_name, chapter_index), notes in sorted(all_notes.items()):
        write_chapter_notes(output_dir, canonical_name, chapter_index, notes)
        print(f"  {canonical_name} / ch{chapter_index:03d}: {len(notes)} notes")


# ---------------------------------------------------------------------------
# Soul doc finalization
# ---------------------------------------------------------------------------


async def finalize_soul_doc(
    canonical_name: str,
    souls_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    api_key: str,
) -> None:
    """Generate a full structured SOUL.md for one character from all available material."""
    seed_path = souls_dir / f"{canonical_name}.md"
    seed_doc = seed_path.read_text(encoding="utf-8") if seed_path.exists() else "(none)"

    insights_path = output_dir / canonical_name / "insights.md"
    insights = insights_path.read_text(encoding="utf-8") if insights_path.exists() else "(none)"

    all_notes: list[dict] = []
    for notes_file in sorted((output_dir / canonical_name).glob("ch*.json")):
        try:
            all_notes.extend(json.loads(notes_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    all_notes.sort(key=lambda n: (n.get("chapter_index", 0), n.get("scene_index", 0)))

    prompt = (
        f"# Character: {canonical_name}\n\n"
        f"## Draft Soul Doc\n\n{seed_doc}\n\n"
        f"## Accumulated Insights\n\n{insights}\n\n"
        f"## All Scene Extraction Notes ({len(all_notes)} scenes)\n\n"
        + json.dumps(all_notes, ensure_ascii=False, indent=2)
    )

    agent = build_agent(
        model_name=args.model,
        provider_name=args.provider,
        api_key=api_key,
        base_url=args.base_url,
        output_type=SoulDocOutput,
        system_prompt=FINALIZATION_SYSTEM_PROMPT,
        thinking=args.thinking,
        output_mode=args.output_mode,
    )

    result = await agent_run_with_retry(agent, prompt)
    soul_doc_content: str = result.output.content

    souls_dir.mkdir(parents=True, exist_ok=True)
    safe_name = canonical_name.replace("/", "_").replace("\\", "_").replace("\0", "")
    out_path = souls_dir / f"{safe_name}.md"
    out_path.write_text(soul_doc_content, encoding="utf-8")
    print(f"  Wrote finalized soul doc: {out_path}")


# ---------------------------------------------------------------------------
# Target character resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-scene character notes using batched LLM processing, "
            "with soul doc accumulation and finalization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m extract.character material/processed/novel/CPK/scenes\n"
            "  uv run python -m extract.character material/processed/novel/CPK/scenes \\\n"
            "      --characters 酒寄彩葉 --model anthropic/claude-opus-4-6\n"
        ),
    )
    parser.add_argument(
        "scenes_base_dir",
        type=Path,
        help="Base directory containing ch{N}/ scene subdirectories",
    )
    parser.add_argument(
        "--aliases-file",
        type=Path,
        default=None,
        help="Path to character_aliases.json (default: <scenes_base_dir>/../character_aliases.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for character notes (default: <scenes_base_dir>/../characters/)",
    )
    parser.add_argument(
        "--souls-dir",
        type=Path,
        default=None,
        help="Soul docs directory (default: <scenes_base_dir>/../souls/)",
    )
    parser.add_argument(
        "--characters",
        nargs="+",
        default=None,
        help="Target character canonical names (default: all characters in aliases file)",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=32000,
        help="Target token count per scene batch (default: 32000)",
    )
    parser.add_argument(
        "--cjk-ratio",
        type=float,
        default=0.65,
        help="Token estimation ratio: tokens ≈ len(text) × cjk_ratio (default: 0.65)",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Skip the soul doc finalization pass",
    )
    add_model_args(parser)
    args = parser.parse_args()

    scenes_base_dir: Path = args.scenes_base_dir
    if not scenes_base_dir.is_dir():
        print(f"Error: {scenes_base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    aliases_file: Path = (
        args.aliases_file or (scenes_base_dir.parent / "character_aliases.json")
    )
    if not aliases_file.exists():
        print(
            f"Error: {aliases_file} not found. Run extract.resolve_aliases first.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir: Path = args.output_dir or (scenes_base_dir.parent / "characters")
    souls_dir: Path = args.souls_dir or (scenes_base_dir.parent / "souls")

    aliases = load_aliases(aliases_file)

    if args.characters:
        target_characters = args.characters
    else:
        target_characters = list(aliases.keys())
    api_key = resolve_api_key(args)

    print(f"Target characters: {target_characters}")
    print(f"Model: {args.model}  (provider: {args.provider})")
    print(f"Output dir: {output_dir}")

    all_scenes = load_all_scenes(scenes_base_dir, aliases, target_characters)
    chapter_count = len({s.chapter for s in all_scenes})
    print(f"Loaded {len(all_scenes)} scenes from {chapter_count} chapters.")

    batches = build_batches(all_scenes, args.batch_tokens, args.cjk_ratio)
    print(
        f"Split into {len(batches)} batches "
        f"(target: {args.batch_tokens} tokens, cjk_ratio: {args.cjk_ratio})."
    )

    async def run() -> None:
        await process_all_batches(
            batches=batches,
            target_characters=target_characters,
            souls_dir=souls_dir,
            output_dir=output_dir,
            args=args,
            api_key=api_key,
        )

        if not args.no_finalize:
            print("\nFinalizing soul docs ...", file=sys.stderr)
            for canonical_name in target_characters:
                print(f"  Finalizing 「{canonical_name}」 ...", file=sys.stderr)
                await finalize_soul_doc(
                    canonical_name, souls_dir, output_dir, args, api_key
                )

    asyncio.run(run())
    print("\nDone.")


if __name__ == "__main__":
    main()
