"""Per-scene character extraction pipeline: generate character notes and finalize soul docs.

Usage:
    uv run python -m yorishiro.character --project <project_dir> --source <source_id>
        [--characters NAME [NAME ...]]
        [--batch-tokens N]
        [--model MODEL] [--provider PROVIDER] [--api-key-env VAR]
        [--base-url URL] [--thinking {none,low,medium,high}]
        [--output-mode {tool,native,prompted}]

Example:
    uv run python -m yorishiro.character --project projects/CPK --source cpk-novel \\
        --characters 酒寄彩葉 かぐや --model anthropic/claude-opus-4-6

Input:
    <scenes_base_dir>/ -- scene text files and manifests (produced by the novel.scenes task)
    character_aliases.json -- canonical-name → alias occurrence mapping
    souls/*.md -- seed soul docs (produced by the novel.aliases task)

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
from typing import Any

from pydantic import BaseModel, Field, create_model
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from yorishiro.agent_utils import add_model_args, build_agent_from_args, estimate_tokens
from yorishiro.backup import ProjectBackup
from yorishiro.project import Project


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


class ComfortMechanism(BaseModel):
    """A self-soothing or comfort-seeking behavior."""

    trigger: str = Field(description="What triggers this behavior (situation, emotion, stressor).")
    action: str = Field(description="What the character does to comfort/soothe themselves.")
    sensory_details: list[str] = Field(
        default_factory=list,
        description="Sensory elements involved (scents, sounds, textures, visuals).",
    )


class RepeatedExpression(BaseModel):
    """A phrase or expression the character uses repeatedly."""

    phrase: str = Field(description="The exact phrase or expression.")
    context: str = Field(description="When/why this phrase is used (deflection, excitement, dismissive, etc.).")
    frequency: str = Field(description="How often: 'always', 'frequently', 'occasionally', or 'rarely'.")


class CharacterSceneNote(BaseModel):
    """Structured extraction of one character's information in one scene."""

    character: str = Field(
        description="Canonical name of the character (must match a target character).",
    )
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
    comfort_mechanisms: list[ComfortMechanism] = Field(
        default_factory=list,
        description="Self-soothing behaviors: what triggers them, what actions, sensory details.",
    )
    repeated_expressions: list[RepeatedExpression] = Field(
        default_factory=list,
        description="Phrases used repeatedly by this character in this scene.",
    )
    persona_shifts: list[str] = Field(
        default_factory=list,
        description="Moments where the character shifts from one persona to another (e.g., formal→informal, composed→emotional).",
    )
    sensory_triggers: list[str] = Field(
        default_factory=list,
        description="Sensory inputs that evoke emotional responses (specific songs, scents, objects, places).",
    )

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
            "Cross-scene consistent behavioral patterns. MUST include when observed: "
            "(1) Physical rituals with sensory details (e.g., 'listens to X song when stressed', "
            "'uses specific scent for sleep'), (2) Deflection patterns (phrases or behaviors "
            "used to brush off praise, emotions, difficult topics), (3) Self-soothing mechanisms, "
            "(4) Habits repeated across multiple scenes. Each pattern MUST have scene references."
        ),
    )
    simulation_directives: list[str] = Field(
        default_factory=list,
        description=(
            "Meta-instructions for roleplay simulation. Include: "
            "(1) timeline-specific behavior rules (how behavior differs across story phases), "
            "(2) interaction patterns (input type → typical response), "
            "(3) prohibited behaviors with context (in situation X, never do Y), "
            "(4) tone/register rules (when to use formal/informal), "
            "(5) fallback strategies when uncertain."
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
        description=(
            "Character arc development. MUST identify: "
            "(1) major timeline phases (e.g., high school vs 10 years later), "
            "(2) phase-specific behavior differences (language, relationships, motivation), "
            "(3) key turning points between phases."
        ),
    )


def make_extraction_models(canonical_names: list[str]) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create dynamically constrained CharacterSceneNote and BatchExtractionResult models.
    
    Returns a tuple of (CharacterSceneNote, BatchExtractionResult) with the character
    field constrained to valid canonical names via JSON schema enum.
    """
    # Create CharacterSceneNote with constrained character field
    character_field = Field(
        description="Canonical name of the character (must match a target character).",
        json_schema_extra={"enum": canonical_names},
    )
    
    DynamicCharacterSceneNote = create_model(
        "CharacterSceneNote",
        character=(str, character_field),
        chapter_index=(int, ...),
        scene_index=(int, ...),
        source=(str, "novel"),
        active_persona=(str, ...),
        dialogue_samples=(list[str], ...),
        language_traits=(str, ...),
        emotional_state=(str, ...),
        inferred_motivation=(str, ...),
        internal_conflict=(str, ...),
        relationships=(list[Relationship], ...),
        actions_taken=(str, ...),
        actions_avoided=(str, ...),
        decision_logic=(str, ...),
        arc_marker=(str, ...),
        knowledge_scope=(KnowledgeScope, ...),
        comfort_mechanisms=(list[ComfortMechanism], Field(default_factory=list)),
        repeated_expressions=(list[RepeatedExpression], Field(default_factory=list)),
        persona_shifts=(list[str], Field(default_factory=list)),
        sensory_triggers=(list[str], Field(default_factory=list)),
        __base__=BaseModel,
    )
    DynamicCharacterSceneNote.__doc__ = "Structured extraction of one character's information in one scene."

    # Create BatchExtractionResult referencing the dynamic CharacterSceneNote
    DynamicBatchExtractionResult = create_model(
        "BatchExtractionResult",
        notes=(list[DynamicCharacterSceneNote], Field(  # type: ignore
            description=(
                "One CharacterSceneNote per (target character × scene) where the character "
                "actually appears. Identified by character + chapter_index + scene_index. "
                "Omit entirely for characters not present in a scene."
            )
        )),
        soul_doc_appends=(list[SoulDocAppend], Field(
            default_factory=list,
            description=(
                "One SoulDocAppend per target character with genuinely new insights from "
                "this batch. Only include characters with something new to add. "
                "Do NOT repeat content already in the character's soul doc or insights."
            ),
        )),
        __base__=BaseModel,
    )
    DynamicBatchExtractionResult.__doc__ = "LLM output for one scene batch."
    
    return DynamicCharacterSceneNote, DynamicBatchExtractionResult


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
- `comfort_mechanisms`: Self-soothing behaviors with trigger, action, and sensory details. \
Look for: listening to specific music when stressed, using specific scents for sleep, \
repetitive physical actions (checking locks, counting), escapist activities.
- `repeated_expressions`: Phrases this character uses repeatedly — catchphrases, \
deflection phrases, nervous tics in speech. Include the phrase itself, context of use, \
and frequency (always/frequently/occasionally/rarely).
- `persona_shifts`: Moments where the character switches from one mode to another \
(e.g., formal customer service mode → casual with friends, composed→breaking down).
- `sensory_triggers`: Sensory inputs that evoke emotional responses — specific songs, \
scents, objects, places, visual motifs that have emotional significance.

### B. SoulDocAppend (per character, synthesized across the batch)

For each target character appearing in this batch, synthesize cross-scene insights that:
1. Are relevant to a final SOUL.md document (negative constraints, behavioral patterns, \
simulation directives, relationship history, personality synthesis, arc development)
2. Are **not** already captured by the per-scene CharacterSceneNote fields above
3. Are **not** already present in the character's soul doc or accumulated insights

Omit a character from soul_doc_appends if there is genuinely nothing new to add.

**Traceability**: Append scene identifiers to each insight showing which scenes it came from:
- "Never directly states feelings (ch003/s02, ch003/s05)"
- "Uses formal speech with strangers (ch001/s01, ch003/s02)"

**Deduplication**: Do NOT repeat the same insight multiple times. If multiple scenes in this \
batch show the same behavioral pattern, combine them into one insight listing all relevant scenes:
- GOOD: "Uses formal speech with strangers (ch001/s01, ch003/s02)"
- BAD: Two separate lines with identical content

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


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def agent_run_with_retry(agent, prompt: str, max_attempts: int = 3):
    """Run agent, retrying on output validation errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await agent.run(prompt)
        except (ValidationError, UnexpectedModelBehavior, json.JSONDecodeError) as exc:
            last_exc = exc
            print(
                f"  [Output validation error attempt {attempt}/{max_attempts}] {exc} — retrying ...",
                file=sys.stderr,
            )
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Token estimation / batching
# ---------------------------------------------------------------------------


def build_batches(
    scenes: list[SceneRecord],
    batch_tokens: int,
) -> list[list[SceneRecord]]:
    """Group scenes into batches not exceeding batch_tokens estimated tokens."""
    batches: list[list[SceneRecord]] = []
    current_batch: list[SceneRecord] = []
    current_tokens = 0
    for scene in scenes:
        scene_tokens = estimate_tokens(scene.text)
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


def load_insight_context(canonical_name: str, characters_dir: Path) -> str:
    """Return accumulated insights text for a character."""
    insights_path = characters_dir / canonical_name / "insights.md"
    if insights_path.exists():
        return f"### Accumulated Insights\n\n{insights_path.read_text(encoding='utf-8')}"
    return "(no background available yet)"


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
    chapter_stem: str,
    notes: list[dict],
) -> Path:
    out_path = output_dir / canonical_name / f"{chapter_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notes_sorted = sorted(notes, key=lambda n: n.get("scene_index", 0))
    out_path.write_text(
        json.dumps(notes_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


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
    characters_dir: Path,
    output_dir: Path,
    agent: Any,
) -> None:
    total = len(batches)
    # chapter_index (int) → chapter directory stem (str), e.g. 3 → "ch003"
    chapter_index_to_stem: dict[int, str] = {
        scene.chapter_index: scene.chapter
        for batch in batches
        for scene in batch
    }
    # Accumulate notes keyed by (canonical_name, chapter_index)
    all_notes: dict[tuple[str, int], list[dict]] = {}

    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{total}  ({len(batch)} scenes) ...", file=sys.stderr)

        # Only load insight contexts for target characters appearing in this batch
        chars_in_batch = {c.canonical for scene in batch for c in scene.target_characters}
        insight_contexts = {
            name: load_insight_context(name, characters_dir)
            for name in target_characters
            if name in chars_in_batch
        }

        prompt = format_batch_prompt(batch, insight_contexts, i, total)
        bg_chars = sum(len(ctx) for ctx in insight_contexts.values())
        scene_chars = sum(len(s.text) for s in batch)
        print(
            f"  Prompt sizes: backgrounds={bg_chars} chars, scenes={scene_chars} chars, "
            f"total={len(prompt)} chars",
            file=sys.stderr,
        )
        result = await agent_run_with_retry(agent, prompt)
        extraction = result.output

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
        chapter_stem = chapter_index_to_stem.get(chapter_index, f"ch{chapter_index:03d}")
        write_chapter_notes(output_dir, canonical_name, chapter_stem, notes)
        print(f"  {canonical_name} / {chapter_stem}: {len(notes)} notes")


# ---------------------------------------------------------------------------
# Target character resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-scene character notes using batched LLM processing.\n"
            "Final SOUL.md generation is handled by yorishiro.synthesize."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.character --project projects/CPK --source cpk-novel\n"
            "  uv run python -m yorishiro.character --source cpk-novel \\\n"
            "      --characters 酒寄彩葉 --model anthropic/claude-opus-4-6\n"
        ),
    )
    
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source ID within project",
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
        "--no-backup",
        action="store_true",
        help="Skip backup snapshot after processing",
    )
    add_model_args(parser)
    args = parser.parse_args()

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)
    
    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)
    
    config = project.resolved_model_config("character")
    
    scenes_base_dir = project.source_dir(args.source) / "scenes"
    characters_dir = project.source_dir(args.source) / "characters"
    aliases_file = characters_dir / "character_aliases.json"
    output_dir = characters_dir
    
    if not aliases_file.exists():
        print(
            f"Error: {aliases_file} not found. Run the novel.aliases task first.",
            file=sys.stderr,
        )
        sys.exit(1)

    aliases = load_aliases(aliases_file)

    if args.characters:
        target_characters = args.characters
    else:
        target_characters = list(aliases.keys())

    model_display = args.model or config.name or "unknown"
    provider_display = args.provider or config.provider or "unknown"
    print(f"Target characters: {target_characters}")
    print(f"Model: {model_display}  (provider: {provider_display})")
    print(f"Output dir: {output_dir}")

    all_scenes = load_all_scenes(scenes_base_dir, aliases, target_characters)
    chapter_count = len({s.chapter for s in all_scenes})
    print(f"Loaded {len(all_scenes)} scenes from {chapter_count} chapters.")

    batches = build_batches(all_scenes, args.batch_tokens)
    print(f"Split into {len(batches)} batches (target: {args.batch_tokens} tokens).")

    backup = ProjectBackup(project.root)

    async def run() -> None:
        _, BatchExtractionResult = make_extraction_models(target_characters)
        extraction_agent = build_agent_from_args(
            args,
            output_type=BatchExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            config=config,
        )

        await process_all_batches(
            batches=batches,
            target_characters=target_characters,
            characters_dir=characters_dir,
            output_dir=output_dir,
            agent=extraction_agent,
        )

    asyncio.run(run())

    if not args.no_backup:
        backup.snapshot(f"character-{args.source}")

    print("\nDone.")


if __name__ == "__main__":
    main()
