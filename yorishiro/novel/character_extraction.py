"""Character extraction engine for novel scene batches."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from yorishiro.agent_utils import estimate_tokens


class Relationship(BaseModel):
    target: str
    dynamic: str
    shift: str | None = None


class KnowledgeScope(BaseModel):
    facts_revealed: list[str]
    facts_hidden: list[str]


class ComfortMechanism(BaseModel):
    """A self-soothing or comfort-seeking behavior."""

    trigger: str = Field(
        description="What triggers this behavior (situation, emotion, stressor)."
    )
    action: str = Field(
        description="What the character does to comfort/soothe themselves."
    )
    sensory_details: list[str] = Field(
        default_factory=list,
        description="Sensory elements involved (scents, sounds, textures, visuals).",
    )


class RepeatedExpression(BaseModel):
    """A phrase or expression the character uses repeatedly."""

    phrase: str = Field(description="The exact phrase or expression.")
    context: str = Field(
        description="When/why this phrase is used (deflection, excitement, dismissive, etc.)."
    )
    frequency: str = Field(
        description="How often: 'always', 'frequently', 'occasionally', or 'rarely'."
    )


class CharacterSceneNote(BaseModel):
    """Structured extraction of one character's information in one scene."""

    character: str = Field(
        description="Canonical name of the character (must match a target character)."
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
    comfort_mechanisms: list[ComfortMechanism] = Field(default_factory=list)
    repeated_expressions: list[RepeatedExpression] = Field(default_factory=list)
    persona_shifts: list[str] = Field(default_factory=list)
    sensory_triggers: list[str] = Field(default_factory=list)


class SoulDocAppend(BaseModel):
    """Per-character synthesized insights from a batch, for soul doc accumulation."""

    canonical_name: str = Field(
        description="Canonical name of the character (must match a target character)."
    )
    negative_constraints: list[str] = Field(default_factory=list)
    behavioral_patterns: list[str] = Field(default_factory=list)
    simulation_directives: list[str] = Field(default_factory=list)
    relationship_insights: list[str] = Field(default_factory=list)
    personality_synthesis: list[str] = Field(default_factory=list)
    arc_notes: str = Field(default="")


def make_extraction_models(
    canonical_names: list[str],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create constrained CharacterSceneNote and BatchExtractionResult models."""
    character_field = Field(
        description="Canonical name of the character (must match a target character).",
        json_schema_extra={"enum": canonical_names},  # pyright: ignore[reportArgumentType]
    )

    dynamic_character_scene_note = create_model(
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
    dynamic_character_scene_note.__doc__ = (
        "Structured extraction of one character's information in one scene."
    )

    dynamic_batch_extraction_result = create_model(
        "BatchExtractionResult",
        notes=(
            list[dynamic_character_scene_note],  # ty: ignore[invalid-type-form]
            Field(
                description=(
                    "One CharacterSceneNote per (target character × scene) where the character "
                    "actually appears. Identified by character + chapter_index + scene_index. "
                    "Omit entirely for characters not present in a scene."
                )
            ),
        ),
        soul_doc_appends=(
            list[SoulDocAppend],
            Field(
                default_factory=list,
                description=(
                    "One SoulDocAppend per target character with genuinely new insights from "
                    "this batch. Only include characters with something new to add. "
                    "Do NOT repeat content already in the character's soul doc or insights."
                ),
            ),
        ),
        __base__=BaseModel,
    )
    dynamic_batch_extraction_result.__doc__ = "LLM output for one scene batch."
    return dynamic_character_scene_note, dynamic_batch_extraction_result


@dataclass
class CharacterInScene:
    alias: str
    canonical: str


@dataclass
class SceneRecord:
    chapter: str
    chapter_index: int
    scene_index: int
    location: str
    time: str
    characters: list[CharacterInScene]
    target_characters: list[CharacterInScene]
    text: str


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


class AgentRunResult(Protocol):
    output: BaseModel


class CharacterExtractionAgent(Protocol):
    async def run(self, prompt: str) -> AgentRunResult: ...


async def agent_run_with_retry(
    agent: CharacterExtractionAgent, prompt: str, max_attempts: int = 3
) -> AgentRunResult:
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


def build_batches(
    scenes: list[SceneRecord], batch_tokens: int
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


def load_aliases(aliases_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load character_aliases.json, excluding UNRESOLVED."""
    data = json.loads(aliases_path.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if key != "UNRESOLVED"}


def build_alias_canonical_lookup(
    aliases: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, int, str], str]:
    """Build (chapter, scene_index, alias) → canonical_name lookup."""
    lookup: dict[tuple[str, int, str], str] = {}
    for canonical, occurrences in aliases.items():
        for occurrence in occurrences:
            lookup[
                (occurrence["chapter"], occurrence["scene"], occurrence["alias"])
            ] = canonical
    return lookup


def load_all_scenes(
    scenes_base_dir: Path,
    aliases: dict[str, list[dict[str, Any]]],
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

            characters = [
                CharacterInScene(
                    alias=alias,
                    canonical=alias_lookup.get((chapter, scene_index, alias), alias)
                    or alias,
                )
                for alias in scene.get("characters", [])
            ]
            target_in_scene = [
                character
                for character in characters
                if character.canonical in target_set
            ]

            scenes.append(
                SceneRecord(
                    chapter=chapter,
                    chapter_index=chapter_index,
                    scene_index=scene_index,
                    location=scene.get("location", ""),
                    time=scene.get("time", ""),
                    characters=characters,
                    target_characters=target_in_scene,
                    text=scene_file.read_text(encoding="utf-8"),
                )
            )

    return sorted(scenes, key=lambda scene: (scene.chapter, scene.scene_index))


def load_insight_context(canonical_name: str, characters_dir: Path) -> str:
    """Return accumulated insights text for a character."""
    insights_path = characters_dir / canonical_name / "insights.md"
    if insights_path.exists():
        return (
            f"### Accumulated Insights\n\n{insights_path.read_text(encoding='utf-8')}"
        )
    return "(no background available yet)"


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
        annotations = [
            f"「{character.alias}」→ {character.canonical}"
            if character.alias != character.canonical
            else f"「{character.alias}」"
            for character in scene.characters
        ]
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


def write_chapter_notes(
    output_dir: Path,
    canonical_name: str,
    chapter_stem: str,
    notes: list[dict[str, Any]],
) -> Path:
    out_path = output_dir / canonical_name / f"{chapter_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notes_sorted = sorted(notes, key=lambda note: int(note.get("scene_index", 0)))
    out_path.write_text(
        json.dumps(notes_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def append_insights(
    output_dir: Path, append: SoulDocAppend, batch_num: int, total: int
) -> None:
    """Append batch insights to characters/{name}/insights.md."""
    insights_path = output_dir / append.canonical_name / "insights.md"
    insights_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"\n## Batch {batch_num}/{total}\n"]
    if append.negative_constraints:
        lines.append("### Negative Constraints\n")
        lines.extend(f"- {constraint}" for constraint in append.negative_constraints)
        lines.append("")
    if append.behavioral_patterns:
        lines.append("### Behavioral Patterns\n")
        lines.extend(f"- {pattern}" for pattern in append.behavioral_patterns)
        lines.append("")
    if append.simulation_directives:
        lines.append("### Simulation Directives\n")
        lines.extend(f"- {directive}" for directive in append.simulation_directives)
        lines.append("")
    if append.relationship_insights:
        lines.append("### Relationship Insights\n")
        lines.extend(f"- {insight}" for insight in append.relationship_insights)
        lines.append("")
    if append.personality_synthesis:
        lines.append("### Personality Synthesis\n")
        lines.extend(f"- {synthesis}" for synthesis in append.personality_synthesis)
        lines.append("")
    if append.arc_notes:
        lines.append("### Arc Notes\n")
        lines.append(append.arc_notes)
        lines.append("")

    with insights_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


async def process_all_batches(
    batches: list[list[SceneRecord]],
    target_characters: list[str],
    characters_dir: Path,
    output_dir: Path,
    agent: CharacterExtractionAgent,
) -> None:
    total = len(batches)
    chapter_index_to_stem = {
        scene.chapter_index: scene.chapter for batch in batches for scene in batch
    }
    all_notes: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for batch_num, batch in enumerate(batches, 1):
        print(
            f"Processing batch {batch_num}/{total}  ({len(batch)} scenes) ...",
            file=sys.stderr,
        )
        chars_in_batch = {
            character.canonical
            for scene in batch
            for character in scene.target_characters
        }
        insight_contexts = {
            name: load_insight_context(name, characters_dir)
            for name in target_characters
            if name in chars_in_batch
        }

        prompt = format_batch_prompt(batch, insight_contexts, batch_num, total)
        print(
            f"  Prompt sizes: backgrounds={sum(len(ctx) for ctx in insight_contexts.values())} chars, "
            f"scenes={sum(len(scene.text) for scene in batch)} chars, total={len(prompt)} chars",
            file=sys.stderr,
        )
        result = await agent_run_with_retry(agent, prompt)
        extraction = result.output

        notes = getattr(extraction, "notes")
        soul_doc_appends = getattr(extraction, "soul_doc_appends")

        for note in notes:
            all_notes.setdefault((note.character, note.chapter_index), []).append(
                note.model_dump()
            )

        for app in soul_doc_appends:
            if any(
                [
                    app.negative_constraints,
                    app.behavioral_patterns,
                    app.simulation_directives,
                    app.relationship_insights,
                    app.personality_synthesis,
                    app.arc_notes,
                ]
            ):
                append_insights(output_dir, app, batch_num, total)

        print(
            f"  → {len(notes)} notes, {len(soul_doc_appends)} soul doc appends",
            file=sys.stderr,
        )

    print("\nWriting character notes ...", file=sys.stderr)
    for (canonical_name, chapter_index), notes in sorted(all_notes.items()):
        chapter_stem = chapter_index_to_stem.get(
            chapter_index, f"ch{chapter_index:03d}"
        )
        write_chapter_notes(output_dir, canonical_name, chapter_stem, notes)
        print(f"  {canonical_name} / {chapter_stem}: {len(notes)} notes")
