"""Per-scene character extraction pipeline: generate character notes from scene files.

Usage:
    uv run python -m extract.character <scenes_base_dir>
        [--aliases-file PATH] [--output-dir PATH]
        [--characters NAME [NAME ...]] [--material-yaml PATH]
        [--model MODEL] [--provider PROVIDER] [--api-key-env VAR]
        [--base-url URL] [--thinking {none,low,medium,high}]
        [--output-mode {tool,native,prompted}] [--force]

Example:
    uv run python -m extract.character material/processed/novel/CPK/scenes \\
        --characters 彩葉 輝耀 --model anthropic/claude-opus-4-6

Input:
    <scenes_base_dir>/ -- scene text files and manifests (produced by extract.scene)
    character_aliases.json -- canonical-name → alias occurrence mapping
    material.yaml -- target_characters list (used when --characters not specified)

Output:
    <output_dir>/{canonical_name}/ch{N}.json for each target character × chapter
    Each file is a JSON array of CharacterSceneNote objects (one per scene the character appears in).

Processing:
    One LLM call per scene covers ALL target characters appearing in that scene.
    Scenes where a character doesn't appear produce no note for that character.
    Incremental: skips scenes where all target characters already have notes (unless --force).
    Non-narrative scenes (location="N/A") are always skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from extract.agent_utils import add_model_args, build_agent, resolve_api_key
from extract.character_extractor import (
    SCENE_SYSTEM_PROMPT,
    CharacterSceneNote,
    SceneExtractionResult,
    build_multi_scene_extraction_prompt,
)


# ---------------------------------------------------------------------------
# Alias / scene index helpers
# ---------------------------------------------------------------------------


def load_aliases(aliases_path: Path) -> dict[str, list[dict]]:
    """Load character_aliases.json.

    Returns dict mapping canonical_name -> list of {chapter, scene, alias} dicts.
    The special "UNRESOLVED" key is excluded.
    """
    data = json.loads(aliases_path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "UNRESOLVED"}


def build_scene_index(
    target_characters: list[str],
    aliases: dict[str, list[dict]],
) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Build scene-centric index from character_aliases.json.

    Returns dict mapping (chapter_dir, scene_index) -> [(canonical_name, alias_in_scene)].
    Only includes target characters. alias_in_scene is the exact string from character_aliases.json.
    """
    index: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for canonical_name in target_characters:
        if canonical_name not in aliases:
            continue
        for occ in aliases[canonical_name]:
            key = (occ["chapter"], occ["scene"])
            index[key].append((canonical_name, occ["alias"]))
    return dict(index)


def load_existing_scene_indices(output_path: Path) -> set[int]:
    """Return set of scene_index values already present in an output JSON file."""
    if not output_path.exists():
        return set()
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        return {note["scene_index"] for note in data if "scene_index" in note}
    except (json.JSONDecodeError, TypeError):
        return set()


def load_existing_notes(output_path: Path) -> list[dict]:
    """Load existing notes from an output JSON file, or return empty list."""
    if not output_path.exists():
        return []
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError):
        return []


def get_chapter_index_from_manifest(manifest: dict) -> int:
    return manifest.get("chapter_index", 0)


# ---------------------------------------------------------------------------
# Target character resolution
# ---------------------------------------------------------------------------


def load_target_characters(material_yaml_path: Path) -> list[str]:
    """Load target character names from material.yaml project_context.target_characters."""
    data = yaml.safe_load(material_yaml_path.read_text(encoding="utf-8"))
    chars = data.get("project_context", {}).get("target_characters", [])
    return [c["name"] for c in chars if "name" in c]


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


async def extract_scene_notes(
    scene_characters: list[tuple[str, str]],
    chapter_index: int,
    scene_index: int,
    location: str,
    time: str,
    scene_content: str,
    agent,
) -> list[CharacterSceneNote]:
    """Call LLM for one scene. Returns list of CharacterSceneNote for present characters.

    scene_characters: [(canonical_name, alias_in_scene)] — all target characters for this scene.
    """
    user_prompt = build_multi_scene_extraction_prompt(
        scene_characters=scene_characters,
        chapter_index=chapter_index,
        scene_index=scene_index,
        location=location,
        time=time,
        scene_content=scene_content,
    )
    result = await agent.run(user_prompt)
    extraction: SceneExtractionResult = result.output

    # Ensure chapter_index, scene_index, and character fields are correct
    canonical_names = {alias: canonical for canonical, alias in scene_characters}
    corrected: list[CharacterSceneNote] = []
    for note in extraction.notes:
        # Use canonical name (LLM may have used alias or slightly wrong name)
        canonical = note.character
        if canonical not in {c for c, _ in scene_characters}:
            # Try to find the intended canonical by matching aliases
            matched = canonical_names.get(canonical)
            if matched:
                canonical = matched
        corrected.append(note.model_copy(update={
            "character": canonical,
            "chapter_index": chapter_index,
            "scene_index": scene_index,
        }))
    return corrected


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_chapter_notes(output_path: Path, notes: list[dict]) -> None:
    """Write sorted chapter notes array to JSON file."""
    notes_sorted = sorted(notes, key=lambda n: n.get("scene_index", 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notes_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


async def process_all(
    scenes_base_dir: Path,
    output_dir: Path,
    target_characters: list[str],
    aliases: dict[str, list[dict]],
    agent,
    force: bool,
) -> None:
    scene_index = build_scene_index(target_characters, aliases)

    # Group by chapter for ordered processing
    chapters: dict[str, dict[int, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for (chapter_dir, scene_idx), char_list in scene_index.items():
        chapters[chapter_dir][scene_idx] = char_list

    total_processed = 0
    total_skipped = 0

    for chapter_dir in sorted(chapters.keys()):
        chapter_scenes = chapters[chapter_dir]
        manifest_path = scenes_base_dir / chapter_dir / "scenes_manifest.json"
        if not manifest_path.exists():
            print(f"  Warning: manifest not found for {chapter_dir}, skipping", file=sys.stderr)
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chapter_index = get_chapter_index_from_manifest(manifest)
        scene_meta: dict[int, dict] = {s["scene_index"]: s for s in manifest.get("scenes", [])}

        # Determine which characters already have notes for each scene in this chapter
        char_existing: dict[str, set[int]] = {}
        for canonical in target_characters:
            out_path = output_dir / canonical / f"ch{chapter_index:03d}.json"
            char_existing[canonical] = load_existing_scene_indices(out_path)

        # Accumulate new notes per character
        new_notes: dict[str, list[dict]] = defaultdict(list)

        for scene_idx in sorted(chapter_scenes.keys()):
            char_list = chapter_scenes[scene_idx]
            meta = scene_meta.get(scene_idx, {})

            # Skip non-narrative scenes
            if meta.get("location") == "N/A":
                total_skipped += 1
                continue

            # Skip if all characters for this scene already have notes
            chars_needing_note = [
                (canonical, alias)
                for canonical, alias in char_list
                if force or scene_idx not in char_existing.get(canonical, set())
            ]
            if not chars_needing_note:
                total_skipped += 1
                continue

            # Read scene text
            scene_file = scenes_base_dir / chapter_dir / meta.get("file", f"scene_{scene_idx:03d}.txt")
            if not scene_file.exists():
                print(f"  Warning: scene file not found: {scene_file}", file=sys.stderr)
                continue

            scene_content = scene_file.read_text(encoding="utf-8")
            location = meta.get("location", "")
            time = meta.get("time", "")

            print(f"  {chapter_dir}/scene_{scene_idx:03d}  [{location}]  characters: {[c for c, _ in chars_needing_note]}")

            notes = await extract_scene_notes(
                scene_characters=chars_needing_note,
                chapter_index=chapter_index,
                scene_index=scene_idx,
                location=location,
                time=time,
                scene_content=scene_content,
                agent=agent,
            )

            for note in notes:
                new_notes[note.character].append(note.model_dump())

            total_processed += 1

        # Write updated notes for each character in this chapter
        for canonical in target_characters:
            char_new = new_notes.get(canonical, [])
            if not char_new:
                continue
            out_path = output_dir / canonical / f"ch{chapter_index:03d}.json"
            existing = load_existing_notes(out_path)

            # Merge: keep existing notes whose scene_index is not being overwritten
            new_scene_indices = {n["scene_index"] for n in char_new}
            merged = [n for n in existing if n.get("scene_index") not in new_scene_indices]
            merged.extend(char_new)

            write_chapter_notes(out_path, merged)
            print(f"  Wrote {len(char_new)} new note(s) for 「{canonical}」 → {out_path}")

    print(f"\nDone. Processed {total_processed} scenes, skipped {total_skipped}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-scene character notes using LLM (one call per scene, all characters).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m extract.character material/processed/novel/CPK/scenes\n"
            "  uv run python -m extract.character material/processed/novel/CPK/scenes \\\n"
            "      --characters 彩葉 輝耀 --model anthropic/claude-opus-4-6\n"
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
        "--characters",
        nargs="+",
        default=None,
        help="Target character canonical names to extract (default: all from material.yaml)",
    )
    parser.add_argument(
        "--material-yaml",
        type=Path,
        default=Path("material.yaml"),
        help="Path to material.yaml for target_characters (default: material.yaml)",
    )
    parser.add_argument("--force", action="store_true", help="Re-process already-extracted scenes")
    add_model_args(parser)
    args = parser.parse_args()

    scenes_base_dir: Path = args.scenes_base_dir
    if not scenes_base_dir.is_dir():
        print(f"Error: {scenes_base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    aliases_file: Path = args.aliases_file or (scenes_base_dir.parent / "character_aliases.json")
    if not aliases_file.exists():
        print(f"Error: {aliases_file} not found. Run extract.resolve_aliases first.", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output_dir or (scenes_base_dir.parent / "characters")

    # Determine target characters
    if args.characters:
        target_characters = args.characters
    else:
        if not args.material_yaml.exists():
            print(f"Error: {args.material_yaml} not found. Use --characters to specify targets.", file=sys.stderr)
            sys.exit(1)
        target_characters = load_target_characters(args.material_yaml)
        if not target_characters:
            print("Error: No target_characters found in material.yaml", file=sys.stderr)
            sys.exit(1)

    aliases = load_aliases(aliases_file)
    api_key = resolve_api_key(args)

    print(f"Target characters: {target_characters}")
    print(f"Aliases file: {aliases_file}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {args.model} (provider: {args.provider})")

    agent = build_agent(
        model_name=args.model,
        provider_name=args.provider,
        api_key=api_key,
        base_url=args.base_url,
        output_type=SceneExtractionResult,
        system_prompt=SCENE_SYSTEM_PROMPT,
        thinking=args.thinking,
        output_mode=args.output_mode,
    )

    asyncio.run(
        process_all(
            scenes_base_dir=scenes_base_dir,
            output_dir=output_dir,
            target_characters=target_characters,
            aliases=aliases,
            agent=agent,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
