"""Alias resolution pipeline: derive character_aliases.json from scene manifests.

Usage:
    uv run python -m extract.resolve_aliases <scenes_base_dir> [output_file]
        [--model MODEL] [--provider PROVIDER] [--api-key-env VAR]
        [--base-url URL] [--thinking {none,low,medium,high}]
        [--output-mode {tool,native,prompted}]
        [--batch-tokens N] [--cjk-ratio F]
        [--souls-dir PATH] [--no-seed] [--force]

Example:
    uv run python -m extract.resolve_aliases \\
        material/processed/novel/CPK/scenes \\
        material/processed/novel/CPK/character_aliases.json \\
        --model anthropic/claude-opus-4-6 --thinking medium

Input:
    <scenes_base_dir>/ containing ch{N}/ subdirectories, each with:
        scenes_manifest.json  -- scene metadata including characters list
        scene_000.txt, ...    -- scene text files

Output:
    character_aliases.json in the established format:
        {
          "canonical_name": [{"chapter": "ch003", "scene": 0, "alias": "name_as_in_text"}, ...],
          ...
          "UNRESOLVED": [{"chapter": ..., "scene": ..., "alias": ...}, ...]
        }

    souls/<canonical_name>.md  -- initial seed soul doc per character

Process:
    1. Load all scene texts (sorted by chapter / scene index).
    2. Optionally seed GlobalState from existing soul docs in souls/.
    3. Split scenes into batches (~batch_tokens tokens each).
    4. For each batch: send scene texts + metadata + current global state to LLM;
       receive updated character entries, merge instructions, and updated knowledge summary.
    5. Apply updates to GlobalState (occurrence map, character registry).
    6. Validate that all alias names in scene metadata are covered.
    7. Write character_aliases.json and per-character seed soul docs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from extract.agent_utils import add_model_args, build_agent, resolve_api_key


# ---------------------------------------------------------------------------
# Pydantic models for LLM I/O
# ---------------------------------------------------------------------------


class RefutedBelief(BaseModel):
    belief: str = Field(description="What was previously believed.")
    reason: str = Field(description="What disproved it (scene, evidence, or revelation).")


class AliasAssignment(BaseModel):
    """Explicit mapping of one alias occurrence to a canonical character."""

    chapter: str = Field(description="Chapter directory name, e.g. 'ch003'.")
    scene_index: int = Field(description="Scene index within the chapter (0-based).")
    alias: str = Field(
        description="Exact alias string as it appears in the scene metadata."
    )
    canonical_name: str = Field(
        description="Canonical name of the character this alias refers to IN THIS SCENE."
    )


class CharacterEntry(BaseModel):
    """A character's current registry entry."""

    canonical_name: str = Field(
        description=(
            "Primary/canonical name for this character as it appears most prominently "
            "in the source material."
        )
    )
    aliases: list[str] = Field(
        description=(
            "Stable, unambiguous name strings for this character (proper names, fixed titles). "
            "Do NOT include context-dependent references like pronouns or 'the girl' — "
            "those are handled via alias_assignments. Must include canonical_name itself."
        )
    )
    possible_merge_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical names of OTHER registry entries that might refer to the same person "
            "with low confidence. Do not include self."
        ),
    )
    merge_notes: str = Field(
        default="",
        description="Why the possible_merge_candidates might be the same person.",
    )
    known_facts: list[str] = Field(
        default_factory=list,
        description="Concise bullet-point facts about this character accumulated so far.",
    )
    current_state: str = Field(
        default="",
        description=(
            "Character's last known state: location, emotional state, arc position, etc."
        ),
    )
    refuted_beliefs: list[RefutedBelief] = Field(
        default_factory=list,
        description=(
            "Things previously believed about this character that are now known to be false, "
            "each with the reason/evidence that disproved it."
        ),
    )
    extra_notes: str = Field(
        default="",
        description=(
            "Any other relevant notes (composite persona, special narrative role, etc.)."
        ),
    )


class MergeInstruction(BaseModel):
    """Instruction to merge two registry entries into one."""

    keep: str = Field(
        description=(
            "Canonical name to retain. The synthesized merged entry must also appear "
            "in updated_characters under this name."
        )
    )
    absorb: str = Field(
        description="Canonical name being absorbed (will be deleted from the registry).",
    )
    reason: str = Field(
        description="Why these two entries are the same person/persona.",
    )


class UnresolvedAlias(BaseModel):
    """An alias whose identity could not be determined even after reading the scene."""

    chapter: str = Field(description="Chapter directory name, e.g. 'ch003'.")
    scene_index: int = Field(description="Scene index within the chapter (0-based).")
    alias: str = Field(description="The alias string that could not be resolved.")
    reason: str = Field(description="Why the identity could not be determined.")


class BatchUpdateResult(BaseModel):
    """LLM output for one scene batch."""

    updated_characters: list[CharacterEntry] = Field(
        description=(
            "Characters to add or update. Include all characters appearing in this batch, "
            "plus the 'keep' entry (with synthesized facts) for each merge. "
            "Do NOT include characters unaffected by this batch."
        )
    )
    alias_assignments: list[AliasAssignment] = Field(
        description=(
            "Explicit per-scene mapping of every alias name in this batch to its canonical "
            "character. Must cover every (chapter, scene_index, alias) triple listed in "
            "'Characters in scene' metadata — unless listed in unresolved_aliases instead. "
            "The same alias string in different scenes may map to different characters — "
            "always resolve based on scene context."
        )
    )
    unresolved_aliases: list[UnresolvedAlias] = Field(
        default_factory=list,
        description=(
            "Aliases that genuinely cannot be resolved to any character even after reading "
            "the scene. Every alias in 'Characters in scene' metadata must appear in either "
            "alias_assignments OR unresolved_aliases — never omitted from both."
        ),
    )
    merges: list[MergeInstruction] = Field(
        default_factory=list,
        description=(
            "High-confidence identity merges. For each merge, the 'keep' entry with "
            "synthesized facts must also be in updated_characters."
        ),
    )
    knowledge_summary: str = Field(
        description=(
            "Complete replacement for the current knowledge summary. "
            "Preserve all prior content and add new findings from this batch."
        )
    )
    batch_notes: str = Field(
        default="",
        description=(
            "Observations about this batch: non-sequential chapters, side stories, "
            "flashbacks, timeline anomalies, etc."
        ),
    )


class MissedAliasResolution(BaseModel):
    """LLM output for a retry covering only missed aliases."""

    alias_assignments: list[AliasAssignment] = Field(
        description="Assignments for the missed aliases listed in the prompt.",
    )
    unresolved_aliases: list[UnresolvedAlias] = Field(
        default_factory=list,
        description="Any of the missed aliases that still cannot be resolved.",
    )


class SeedFromSoulDocsResult(BaseModel):
    """Output when parsing existing soul docs into registry entries."""

    characters: list[CharacterEntry] = Field(
        description="Character entries parsed from the provided soul documents.",
    )


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class SceneRecord:
    chapter: str
    scene_index: int
    characters: list[str]
    location: str
    time: str
    text: str


@dataclass
class GlobalState:
    characters: dict[str, CharacterEntry] = field(default_factory=dict)
    knowledge_summary: str = ""
    # (chapter, scene_index, alias_string) -> canonical_name
    occurrence_map: dict[tuple[str, int, str], str] = field(default_factory=dict)
    # (chapter, scene_index, alias_string) -> reason string  (LLM-declared unresolvable)
    unresolved_map: dict[tuple[str, int, str], str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BATCH_SYSTEM_PROMPT = """\
You are an expert novel analyst maintaining a living character database and story knowledge base.

## Your Task

You receive:
1. **Current Global State**: A character registry (canonical entries with stable aliases, facts, \
state) and a knowledge summary representing everything understood so far.
2. **Scene Batch**: A set of scene texts with chapter/scene metadata (location, time, \
characters listed in scene manifest).

For each batch, update the global state:

1. **Assign aliases explicitly (per scene)**: For every name listed in `characters_in_scene` \
metadata, output an `AliasAssignment` mapping it to the correct canonical character for THAT \
specific scene. The same string (e.g. "the girl", "彼女") may refer to different characters in \
different scenes — always resolve from context, never assume globally.

2. **Merge with high confidence**: If two registry entries unmistakably refer to the same \
person/persona, merge them. Put the synthesized merged entry in `updated_characters` under \
`keep`'s name; add a `MergeInstruction` declaring which entry to absorb.

3. **Flag possible merges (low confidence)**: If two names might be the same person but you're \
not certain, add the other canonical name to `possible_merge_candidates` with a note — do NOT \
merge yet.

4. **Update canonical names**: If scenes reveal a more official name, update `canonical_name` \
(keep the old name in `aliases`).

5. **Create new entries**: For every new character not yet in the registry — unless they are \
clearly one-off background extras who will never appear again.

6. **Update character knowledge**: Update `known_facts`, `current_state`, `refuted_beliefs`. \
If a prior belief is proven false, move it to `refuted_beliefs` and remove it from `known_facts`.

7. **Update story knowledge**: Update `knowledge_summary` with plot developments, \
world-building, timeline information, relationship changes.

## Critical Rules

- Scenes may be **non-sequential in time** (flashbacks, side stories, parallel timelines) — \
always note this in `batch_notes` when observed.
- Every alias in `characters_in_scene` metadata MUST appear in either `alias_assignments` OR \
`unresolved_aliases` — never omit one from both. Use scene text to resolve ambiguous references \
(pronouns, generic descriptors). If genuinely unresolvable, declare it in `unresolved_aliases` \
with a reason rather than silently skipping it.
- `CharacterEntry.aliases` should only contain stable, unambiguous name strings (proper names, \
fixed titles). Do NOT put pronouns or generic descriptors there.
- Only include characters in `updated_characters` if they appear in this batch OR are affected \
by a merge. Leave untouched characters out.
- `knowledge_summary` is a **complete replacement** — preserve all prior content plus new additions.
- `known_facts` should be concise bullets; remove duplicates; do not repeat facts in \
`refuted_beliefs`.
- **Preserve original-language names**: Use names exactly as they appear in the source text. \
Do NOT translate, romanize, or convert names to another language or writing system.
- **Write content in the source language**: All character data fields (`known_facts`, \
`current_state`, `refuted_beliefs`, `extra_notes`, `merge_notes`, `knowledge_summary`, \
`batch_notes`) must be written in the same language as the source material (Japanese, Chinese, \
etc.). Do NOT translate content into English.
"""

RETRY_SYSTEM_PROMPT = """\
You previously processed a batch of scenes but some aliases were neither assigned nor declared \
unresolved. Provide alias assignments or unresolved declarations for the missed aliases listed \
in the prompt. Every missed alias must end up in alias_assignments or unresolved_aliases.
"""

SEED_SYSTEM_PROMPT = """\
You are parsing existing character soul documents into structured character registry entries.

Each soul document describes one character. Extract into a CharacterEntry:
- canonical_name: the character's primary name
- aliases: all name variations/aliases mentioned
- known_facts: key facts about the character (concise bullets)
- current_state: last known state/arc position
- refuted_beliefs: any explicitly corrected misconceptions
- extra_notes: composite persona notes, special roles, etc.

Parse all provided documents and return one CharacterEntry per character.
"""


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def load_all_scenes(scenes_base_dir: Path) -> list[SceneRecord]:
    """Load all scenes from scenes_manifest.json files, sorted by chapter then scene_index."""
    scenes: list[SceneRecord] = []

    for manifest_path in sorted(scenes_base_dir.glob("*/scenes_manifest.json")):
        chapter = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for scene in manifest.get("scenes", []):
            scene_index = scene["scene_index"]
            scene_file = manifest_path.parent / f"scene_{scene_index:03d}.txt"
            if not scene_file.exists():
                print(
                    f"  Warning: {scene_file} not found, skipping",
                    file=sys.stderr,
                )
                continue

            characters = list(scene.get("characters", []))

            scenes.append(
                SceneRecord(
                    chapter=chapter,
                    scene_index=scene_index,
                    characters=characters,
                    location=scene.get("location", ""),
                    time=scene.get("time", ""),
                    text=scene_file.read_text(encoding="utf-8"),
                )
            )

    return sorted(scenes, key=lambda s: (s.chapter, s.scene_index))


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, cjk_ratio: float) -> int:
    """Rough token estimate: tokens ≈ len(text) * cjk_ratio."""
    return int(len(text) * cjk_ratio)


def build_batches(
    scenes: list[SceneRecord],
    batch_tokens: int,
    cjk_ratio: float = 0.65,
) -> list[list[SceneRecord]]:
    """Group scenes into batches not exceeding batch_tokens estimated tokens.

    Scenes are never split across batches. Each scene is added to the current
    batch until the token budget would be exceeded, then a new batch is started.
    """
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
# Prompt formatting
# ---------------------------------------------------------------------------


def format_global_state(state: GlobalState) -> str:
    """Compact text serialisation of global state for inclusion in batch prompts."""
    lines: list[str] = []

    if state.knowledge_summary:
        lines.append("### Knowledge Summary\n")
        lines.append(state.knowledge_summary)
        lines.append("")
    else:
        lines.append("### Knowledge Summary\n(empty — first batch)\n")

    lines.append(f"### Character Registry ({len(state.characters)} entries)\n")

    for entry in state.characters.values():
        lines.append(f"**{entry.canonical_name}**")
        other_aliases = [a for a in entry.aliases if a != entry.canonical_name]
        if other_aliases:
            lines.append(f"  Aliases: {', '.join(f'「{a}」' for a in other_aliases)}")
        if entry.known_facts:
            lines.append("  Facts: " + " | ".join(entry.known_facts))
        if entry.current_state:
            lines.append(f"  State: {entry.current_state}")
        if entry.possible_merge_candidates:
            lines.append(
                f"  Possible same as: {', '.join(entry.possible_merge_candidates)}"
                + (f" — {entry.merge_notes}" if entry.merge_notes else "")
            )
        if entry.refuted_beliefs:
            lines.append(
                "  Refuted: "
                + " | ".join(f"{rb.belief} ({rb.reason})" for rb in entry.refuted_beliefs)
            )
        if entry.extra_notes:
            lines.append(f"  Notes: {entry.extra_notes}")
        lines.append("")

    return "\n".join(lines)


def format_batch_prompt(
    scenes_batch: list[SceneRecord],
    state: GlobalState,
    batch_num: int,
    total_batches: int,
) -> str:
    """Build the user prompt for one scene batch."""
    parts: list[str] = [
        f"# Batch {batch_num}/{total_batches}  ({len(scenes_batch)} scenes)\n",
        "## Current Global State\n",
        format_global_state(state),
        "## Scene Batch\n",
    ]

    for scene in scenes_batch:
        char_list = (
            ", ".join(f"「{c}」" for c in scene.characters)
            if scene.characters
            else "(none)"
        )
        parts.append(
            f"---\n"
            f"**{scene.chapter} / scene_{scene.scene_index:03d}**"
            f"  | Location: {scene.location or '?'}"
            f"  | Time: {scene.time or '?'}"
            f"  | Characters in scene: {char_list}\n\n"
            f"{scene.text}\n"
        )

    parts.append(
        "\n---\n"
        "Please process the batch above and return a BatchUpdateResult. "
        "Every name listed in 'Characters in scene' metadata must be accounted for "
        "in updated_characters (as canonical_name or in aliases)."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------


def apply_result(
    result: BatchUpdateResult | MissedAliasResolution,
    state: GlobalState,
) -> None:
    """Apply any batch result to GlobalState in place (shared by main and retry calls)."""
    if isinstance(result, BatchUpdateResult):
        # Apply updated characters
        for entry in result.updated_characters:
            state.characters[entry.canonical_name] = entry

        # Apply merges
        for merge in result.merges:
            if merge.absorb != merge.keep and merge.absorb in state.characters:
                del state.characters[merge.absorb]
            for key, canonical in list(state.occurrence_map.items()):
                if canonical == merge.absorb:
                    state.occurrence_map[key] = merge.keep
            for entry in state.characters.values():
                if merge.absorb in entry.possible_merge_candidates:
                    entry.possible_merge_candidates.remove(merge.absorb)

        state.knowledge_summary = result.knowledge_summary

        if result.batch_notes:
            print(f"  [Batch notes] {result.batch_notes}", file=sys.stderr)

    # Apply alias assignments (shared by both result types)
    for assignment in result.alias_assignments:
        key = (assignment.chapter, assignment.scene_index, assignment.alias)
        state.occurrence_map[key] = assignment.canonical_name
        # If previously marked unresolved, clear it now
        state.unresolved_map.pop(key, None)

    # Record explicitly unresolved aliases (shared by both result types)
    for ua in result.unresolved_aliases:
        key = (ua.chapter, ua.scene_index, ua.alias)
        if key not in state.occurrence_map:
            state.unresolved_map[key] = ua.reason


def find_missed(
    result: BatchUpdateResult,
    scenes_batch: list[SceneRecord],
    state: GlobalState,
) -> list[tuple[str, int, str]]:
    """Return (chapter, scene_index, alias) triples not in occurrence_map or unresolved_map."""
    missed = []
    for scene in scenes_batch:
        for alias in scene.characters:
            key = (scene.chapter, scene.scene_index, alias)
            if key not in state.occurrence_map and key not in state.unresolved_map:
                missed.append(key)
    return missed


# ---------------------------------------------------------------------------
# Soul doc seeding
# ---------------------------------------------------------------------------


async def agent_run_with_retry(agent, prompt: str, max_attempts: int = 3):
    """Run agent, retrying on ValidationError (model produced invalid schema output)."""
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


async def seed_from_soul_docs(
    souls_dir: Path,
    args: argparse.Namespace,
    api_key: str,
) -> GlobalState:
    """Parse existing soul docs in souls_dir and return a seeded GlobalState."""
    soul_files = sorted(souls_dir.glob("*.md"))
    if not soul_files:
        return GlobalState()

    print(
        f"Seeding from {len(soul_files)} soul docs in {souls_dir} ...",
        file=sys.stderr,
    )

    doc_sections = [
        f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}" for f in soul_files
    ]
    combined = "\n\n".join(doc_sections)
    prompt = (
        f"Parse the following {len(soul_files)} character soul documents "
        f"into CharacterEntry records.\n\n{combined}"
    )

    agent = build_agent(
        model_name=args.model,
        provider_name=args.provider,
        api_key=api_key,
        base_url=args.base_url,
        output_type=SeedFromSoulDocsResult,
        system_prompt=SEED_SYSTEM_PROMPT,
        thinking=args.thinking,
        output_mode=args.output_mode,
    )
    result = await agent_run_with_retry(agent, prompt)

    state = GlobalState()
    for entry in result.output.characters:
        state.characters[entry.canonical_name] = entry

    print(
        f"  Seeded {len(state.characters)} characters from soul docs.",
        file=sys.stderr,
    )
    return state


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def format_retry_prompt(
    missed: list[tuple[str, int, str]],
    scene_lookup: dict[tuple[str, int], SceneRecord],
    state: GlobalState,
) -> str:
    """Build a re-prompt for aliases the model missed in its previous response."""
    parts = [
        "The following aliases were not assigned or declared unresolved in your previous "
        "response. Please resolve each one now.\n",
        "## Current Character Registry\n",
        format_global_state(state),
        "## Missed Aliases and Their Scenes\n",
    ]
    seen_scenes: set[tuple[str, int]] = set()
    for chapter, scene_index, alias in missed:
        scene_key = (chapter, scene_index)
        parts.append(
            f"- **{chapter}/scene_{scene_index:03d}**: 「{alias}」"
        )
        if scene_key not in seen_scenes:
            seen_scenes.add(scene_key)
            scene = scene_lookup.get(scene_key)
            if scene:
                char_list = ", ".join(f"「{c}」" for c in scene.characters)
                parts.append(
                    f"\n  Scene context — Location: {scene.location or '?'}"
                    f" | Time: {scene.time or '?'}"
                    f" | All characters: {char_list}\n\n"
                    f"  {scene.text}\n"
                )
    parts.append(
        "\nProvide alias_assignments and/or unresolved_aliases covering all missed aliases above."
    )
    return "\n".join(parts)


async def process_all_batches(
    batches: list[list[SceneRecord]],
    initial_state: GlobalState,
    args: argparse.Namespace,
    api_key: str,
    max_retries: int = 2,
) -> GlobalState:
    """Process all scene batches sequentially, with retry for missed aliases."""
    state = initial_state
    total = len(batches)

    # Build a flat lookup for quick scene retrieval during retries
    all_scenes_flat = [scene for batch in batches for scene in batch]
    scene_lookup: dict[tuple[str, int], SceneRecord] = {
        (s.chapter, s.scene_index): s for s in all_scenes_flat
    }

    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{total}  ({len(batch)} scenes) ...", file=sys.stderr)

        prompt = format_batch_prompt(batch, state, i, total)
        agent = build_agent(
            model_name=args.model,
            provider_name=args.provider,
            api_key=api_key,
            base_url=args.base_url,
            output_type=BatchUpdateResult,
            system_prompt=BATCH_SYSTEM_PROMPT,
            thinking=args.thinking,
            output_mode=args.output_mode,
        )
        result = await agent_run_with_retry(agent, prompt)
        apply_result(result.output, state)

        # Retry loop for missed aliases
        for retry in range(1, max_retries + 1):
            missed = find_missed(result.output, batch, state)
            if not missed:
                break
            print(
                f"  [Retry {retry}/{max_retries}] {len(missed)} aliases missed, re-prompting ...",
                file=sys.stderr,
            )
            retry_prompt = format_retry_prompt(missed, scene_lookup, state)
            retry_agent = build_agent(
                model_name=args.model,
                provider_name=args.provider,
                api_key=api_key,
                base_url=args.base_url,
                output_type=MissedAliasResolution,
                system_prompt=RETRY_SYSTEM_PROMPT,
                thinking=args.thinking,
                output_mode=args.output_mode,
            )
            retry_result = await agent_run_with_retry(retry_agent, retry_prompt)
            apply_result(retry_result.output, state)

        still_missed = find_missed(result.output, batch, state)
        if still_missed:
            print(
                f"  Warning: {len(still_missed)} aliases still unresolved after retries:",
                file=sys.stderr,
            )
            for ch, si, alias in still_missed[:10]:
                print(f"    {ch}/scene_{si:03d}: 「{alias}」", file=sys.stderr)

        print(
            f"  → {len(state.characters)} characters, "
            f"{len(state.occurrence_map)} assigned, "
            f"{len(state.unresolved_map)} unresolved",
            file=sys.stderr,
        )

    return state


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_character_aliases(
    state: GlobalState,
    output_path: Path,
) -> None:
    """Write character_aliases.json from state.occurrence_map and unresolved_map."""
    # Group occurrences by canonical_name
    by_canonical: dict[str, list[dict]] = {name: [] for name in state.characters}

    for (chapter, scene_index, alias), canonical in state.occurrence_map.items():
        if canonical in by_canonical:
            by_canonical[canonical].append(
                {"chapter": chapter, "scene": scene_index, "alias": alias}
            )

    # Sort each group for deterministic output
    for entries in by_canonical.values():
        entries.sort(key=lambda o: (o["chapter"], o["scene"]))

    # Unresolved: LLM-declared unresolvable aliases
    unresolved: list[dict] = sorted(
        [
            {"chapter": ch, "scene": si, "alias": alias, "reason": reason}
            for (ch, si, alias), reason in state.unresolved_map.items()
        ],
        key=lambda o: (o["chapter"], o["scene"], o["alias"]),
    )

    output: dict = dict(by_canonical)
    output["UNRESOLVED"] = unresolved

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_soul_docs(state: GlobalState, souls_dir: Path) -> None:
    """Write initial seed soul docs for each character in souls_dir."""
    souls_dir.mkdir(parents=True, exist_ok=True)

    for canonical_name, entry in state.characters.items():
        safe_name = (
            canonical_name.replace("/", "_").replace("\\", "_").replace("\0", "")
        )
        doc_path = souls_dir / f"{safe_name}.md"

        lines: list[str] = [f"# {canonical_name}\n"]

        other_aliases = [a for a in entry.aliases if a != canonical_name]
        if other_aliases:
            lines.append("## Aliases\n")
            lines.extend(f"- {a}" for a in other_aliases)
            lines.append("")

        if entry.known_facts:
            lines.append("## Known Facts\n")
            lines.extend(f"- {fact}" for fact in entry.known_facts)
            lines.append("")

        if entry.current_state:
            lines.append("## Current State\n")
            lines.append(entry.current_state)
            lines.append("")

        if entry.refuted_beliefs:
            lines.append("## Refuted Beliefs\n")
            lines.extend(
                f"- {rb.belief}\n  _(Disproved by: {rb.reason})_"
                for rb in entry.refuted_beliefs
            )
            lines.append("")

        if entry.possible_merge_candidates:
            lines.append("## Possible Identity Overlap\n")
            lines.append(
                f"May be same person as: {', '.join(entry.possible_merge_candidates)}"
            )
            if entry.merge_notes:
                lines.append("")
                lines.append(entry.merge_notes)
            lines.append("")

        if entry.extra_notes:
            lines.append("## Notes\n")
            lines.append(entry.extra_notes)
            lines.append("")

        doc_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"Wrote {len(state.characters)} soul doc seeds to {souls_dir}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve character name aliases from scene texts using "
            "batched LLM processing with a global character registry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m extract.resolve_aliases material/processed/novel/CPK/scenes\n"
            "  uv run python -m extract.resolve_aliases material/processed/novel/CPK/scenes \\\n"
            "      material/processed/novel/CPK/character_aliases.json --force\n"
        ),
    )
    parser.add_argument(
        "scenes_base_dir",
        type=Path,
        help="Base directory containing ch{N}/ scene subdirectories",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=None,
        help="Output JSON file (default: <scenes_base_dir>/../character_aliases.json)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
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
        help=(
            "Token estimation ratio: estimated_tokens = len(text) * cjk_ratio. "
            "Default 0.65 ≈ 1.5 chars per token for CJK text."
        ),
    )
    parser.add_argument(
        "--souls-dir",
        type=Path,
        default=None,
        help=(
            "Directory for soul doc seeds "
            "(default: <output_file_dir>/souls/)"
        ),
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding from existing soul docs even if souls-dir exists",
    )
    add_model_args(parser)
    args = parser.parse_args()

    scenes_base_dir: Path = args.scenes_base_dir
    if not scenes_base_dir.is_dir():
        print(f"Error: {scenes_base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path: Path = (
        args.output_file or (scenes_base_dir.parent / "character_aliases.json")
    )
    souls_dir: Path = args.souls_dir or (output_path.parent / "souls")

    if output_path.exists() and not args.force:
        print(f"Skipping: {output_path} already exists (use --force to overwrite)")
        sys.exit(0)

    api_key = resolve_api_key(args)

    print(f"Loading scenes from {scenes_base_dir} ...")
    all_scenes = load_all_scenes(scenes_base_dir)
    if not all_scenes:
        print(
            "No scenes found. Check that scenes_manifest.json files exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    chapter_count = len({s.chapter for s in all_scenes})
    print(f"  Loaded {len(all_scenes)} scenes from {chapter_count} chapters.")

    batches = build_batches(all_scenes, args.batch_tokens, args.cjk_ratio)
    print(
        f"  Split into {len(batches)} batches "
        f"(target: {args.batch_tokens} tokens each, cjk_ratio: {args.cjk_ratio})."
    )

    async def run() -> GlobalState:
        initial_state = GlobalState()
        if not args.no_seed and souls_dir.exists():
            initial_state = await seed_from_soul_docs(souls_dir, args, api_key)
        return await process_all_batches(batches, initial_state, args, api_key)

    print(f"Processing with {args.model} ...")
    state = asyncio.run(run())

    print(f"\nFinal registry: {len(state.characters)} characters")
    for name, entry in state.characters.items():
        alias_count = len([a for a in entry.aliases if a != name])
        suffix = f" (+{alias_count} aliases)" if alias_count else ""
        print(f"  {name}{suffix}")
    if state.unresolved_map:
        print(f"  UNRESOLVED: {len(state.unresolved_map)} alias occurrences")

    print(f"\nWriting {output_path} ...")
    write_character_aliases(state, output_path)

    print(f"Writing soul doc seeds to {souls_dir} ...")
    write_soul_docs(state, souls_dir)

    print("Done.")


if __name__ == "__main__":
    main()
