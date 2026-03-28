"""Alias resolution pipeline: derive character_aliases.json from scene manifests.

Usage:
    uv run python -m extract.resolve_aliases <scenes_base_dir> [output_file]
        [--model MODEL] [--provider PROVIDER] [--api-key-env VAR]
        [--base-url URL] [--thinking {none,low,medium,high}]
        [--output-mode {tool,native,prompted}] [--force]

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

Process:
    1. Collect all unique character names from all scenes_manifest.json files
    2. Present the list to an LLM agent with a read_scene() tool
    3. Agent reads scenes as needed ("read-until-understood") to resolve aliases
    4. Transform the agent's grouping output to character_aliases.json format
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from extract.agent_utils import add_model_args, build_agent, resolve_api_key


# ---------------------------------------------------------------------------
# Pydantic models for agent output
# ---------------------------------------------------------------------------


class CharacterGroup(BaseModel):
    """A canonical character with all alias strings that refer to them."""

    canonical_name: str = Field(
        description=(
            "The character's primary/canonical name as it appears in the source material. "
            "Choose the most common or most 'official' form of the name."
        )
    )
    aliases: list[str] = Field(
        description=(
            "All alias strings (including the canonical name itself) that refer to this character. "
            "Each alias must be an EXACT copy of the string from the input alias list — "
            "do not add annotations, parentheses, or modify the strings in any way."
        )
    )
    notes: str = Field(
        description="Brief reasoning for how you resolved ambiguous aliases, if any."
    )


class AliasResolutionResult(BaseModel):
    """Complete alias resolution output: canonical character groups + unresolved aliases."""

    groups: list[CharacterGroup] = Field(
        description="All resolved canonical character groups."
    )
    unresolved: list[str] = Field(
        description=(
            "Alias strings from the input that could not be resolved to any canonical character "
            "even after reading all available scenes. Must be exact copies of input strings."
        )
    )


# ---------------------------------------------------------------------------
# System prompt for the alias resolution agent
# ---------------------------------------------------------------------------

ALIAS_SYSTEM_PROMPT = """You are a character analyst for a Japanese novel. Your task is to resolve character name aliases.

## Task

You will be given a list of character name strings (aliases) extracted from scene manifests across chapters of a novel. Each alias is exactly as it appears in the scene text.

Your goal: group all aliases that refer to the same character under a single canonical name.

## How to resolve aliases

1. Start with the alias list and their first-appearance locations.
2. For any alias whose identity is unclear, use the read_scene tool to read the scene text where it first appears.
3. If still unclear after reading the first scene, continue reading subsequent scenes where the alias appears.
4. Keep reading until you understand who the alias refers to, or until you have read all available scenes for that alias.
5. Group aliases by the character they refer to.

## CRITICAL: Output format rules

- The `aliases` field of each CharacterGroup must contain EXACT copies of the alias strings from the input list.
- Do NOT add parenthetical notes, annotations, or any modifications to alias strings.
- The canonical_name should be the character's primary name as it appears most prominently in the source text.
- If an alias is also the canonical name, include it in the aliases list.

## Special handling

- "叙述者" (narrator) has already been excluded from the input and should not appear.
- Channel/duo names (e.g., "いろＰ") may refer to a specific character — read scenes to determine.
- First-person pronouns (私, 僕, etc.) may be aliases for specific characters — use context to resolve.
- If an alias genuinely cannot be resolved after reading all available scenes, add it to unresolved.
"""


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@dataclass
class AliasOccurrence:
    chapter: str    # e.g. "ch003"
    scene: int      # scene_index
    alias: str


@dataclass
class AliasInfo:
    alias: str
    first_chapter: str
    first_scene: int
    all_occurrences: list[AliasOccurrence] = field(default_factory=list)


EXCLUDED_ALIASES = {"叙述者"}


def collect_aliases(scenes_base_dir: Path) -> list[AliasInfo]:
    """Walk all scenes_manifest.json files and collect unique character alias info.

    Returns alias infos sorted by first appearance (chapter asc, scene asc).
    """
    alias_map: dict[str, AliasInfo] = {}

    manifest_paths = sorted(scenes_base_dir.glob("*/scenes_manifest.json"))
    for manifest_path in manifest_paths:
        chapter_dir = manifest_path.parent.name  # e.g. "ch003"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for scene in manifest.get("scenes", []):
            scene_index = scene["scene_index"]
            for alias in scene.get("characters", []):
                if alias in EXCLUDED_ALIASES:
                    continue
                occ = AliasOccurrence(chapter=chapter_dir, scene=scene_index, alias=alias)
                if alias not in alias_map:
                    alias_map[alias] = AliasInfo(
                        alias=alias,
                        first_chapter=chapter_dir,
                        first_scene=scene_index,
                    )
                alias_map[alias].all_occurrences.append(occ)

    return sorted(
        alias_map.values(),
        key=lambda a: (a.first_chapter, a.first_scene),
    )


# ---------------------------------------------------------------------------
# Agent construction and invocation
# ---------------------------------------------------------------------------


def build_read_scene_tool(scenes_base_dir: Path):
    """Return a read_scene function capturing scenes_base_dir."""

    def read_scene(chapter: str, scene_index: int) -> str:
        """Read the text content of a scene file from the scenes directory.

        Args:
            chapter: Chapter directory name, e.g. "ch003"
            scene_index: Scene index (0-based integer)

        Returns:
            Scene text content, or an error message if the file is not found.
        """
        scene_file = scenes_base_dir / chapter / f"scene_{scene_index:03d}.txt"
        if not scene_file.exists():
            return f"Scene file not found: {chapter}/scene_{scene_index:03d}.txt"
        return scene_file.read_text(encoding="utf-8")

    return read_scene


def build_user_prompt(alias_infos: list[AliasInfo]) -> str:
    """Build the user prompt listing all aliases with their occurrence metadata."""
    lines = ["## Alias List (sorted by first appearance)\n"]
    for info in alias_infos:
        occurrences_summary = ", ".join(
            f"{o.chapter}/scene_{o.scene}" for o in info.all_occurrences[:5]
        )
        if len(info.all_occurrences) > 5:
            occurrences_summary += f", ... ({len(info.all_occurrences)} total)"
        lines.append(
            f'- alias: 「{info.alias}」  '
            f'first: {info.first_chapter}/scene_{info.first_scene}  '
            f'all: [{occurrences_summary}]'
        )
    lines.append(
        "\nPlease resolve all aliases above into canonical character groups. "
        "Use the read_scene tool to read scene text as needed."
    )
    return "\n".join(lines)


async def resolve_aliases(
    scenes_base_dir: Path,
    alias_infos: list[AliasInfo],
    args: argparse.Namespace,
    api_key: str,
) -> AliasResolutionResult:
    read_scene = build_read_scene_tool(scenes_base_dir)
    agent = build_agent(
        model_name=args.model,
        provider_name=args.provider,
        api_key=api_key,
        base_url=args.base_url,
        output_type=AliasResolutionResult,
        system_prompt=ALIAS_SYSTEM_PROMPT,
        thinking=args.thinking,
        output_mode=args.output_mode,
        tools=[read_scene],
    )
    user_prompt = build_user_prompt(alias_infos)
    result = await agent.run(user_prompt)
    return result.output


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_character_aliases(
    result: AliasResolutionResult,
    alias_infos: list[AliasInfo],
    output_path: Path,
) -> None:
    """Transform AliasResolutionResult to character_aliases.json format and write."""
    # Build lookup: alias_string -> AliasInfo
    alias_lookup: dict[str, AliasInfo] = {info.alias: info for info in alias_infos}

    output: dict[str, list[dict]] = {}

    for group in result.groups:
        occurrences: list[dict] = []
        for alias_str in group.aliases:
            info = alias_lookup.get(alias_str)
            if info is None:
                print(
                    f"  Warning: alias 「{alias_str}」 from LLM output not found in collected data",
                    file=sys.stderr,
                )
                continue
            for occ in info.all_occurrences:
                occurrences.append({
                    "chapter": occ.chapter,
                    "scene": occ.scene,
                    "alias": occ.alias,
                })
        # Sort by chapter then scene for deterministic output
        occurrences.sort(key=lambda o: (o["chapter"], o["scene"]))
        output[group.canonical_name] = occurrences

    # Unresolved entries: include all occurrences for manual review
    unresolved_entries: list[dict] = []
    for alias_str in result.unresolved:
        info = alias_lookup.get(alias_str)
        if info:
            for occ in info.all_occurrences:
                unresolved_entries.append({
                    "chapter": occ.chapter,
                    "scene": occ.scene,
                    "alias": occ.alias,
                })
    output["UNRESOLVED"] = unresolved_entries

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve character name aliases from scene manifests using LLM.",
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
    add_model_args(parser)
    args = parser.parse_args()

    scenes_base_dir: Path = args.scenes_base_dir
    if not scenes_base_dir.is_dir():
        print(f"Error: {scenes_base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path: Path = args.output_file or (scenes_base_dir.parent / "character_aliases.json")

    if output_path.exists() and not args.force:
        print(f"Skipping: {output_path} already exists (use --force to overwrite)")
        sys.exit(0)

    api_key = resolve_api_key(args)

    print(f"Collecting aliases from {scenes_base_dir} ...")
    alias_infos = collect_aliases(scenes_base_dir)
    if not alias_infos:
        print("No aliases found. Check that scenes_manifest.json files exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(alias_infos)} unique aliases. Resolving with {args.model} ...")
    for info in alias_infos:
        print(f"  「{info.alias}」  first: {info.first_chapter}/scene_{info.first_scene}  ({len(info.all_occurrences)} occurrences)")

    result = asyncio.run(resolve_aliases(scenes_base_dir, alias_infos, args, api_key))

    print(f"\nResolution complete: {len(result.groups)} characters, {len(result.unresolved)} unresolved")
    for group in result.groups:
        print(f"  {group.canonical_name}: {group.aliases}")
    if result.unresolved:
        print(f"  UNRESOLVED: {result.unresolved}")

    print(f"\nWriting {output_path} ...")
    write_character_aliases(result, alias_infos, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
