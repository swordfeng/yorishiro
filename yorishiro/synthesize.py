"""Synthesize final SOUL.md documents from accumulated character data.

Usage:
    uv run python -m yorishiro.synthesize --project projects/CPK
    uv run python -m yorishiro.synthesize --project projects/CPK --character 酒寄彩葉

Input:
    For each character, from all sources:
    - characters/{name}/insights.md -- accumulated insights
    - characters/{name}/ch{N}.json -- per-chapter scene notes

Output:
    souls/{name}.md -- final SOUL.md document
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from yorishiro.agent_utils import add_model_args, build_agent_from_args
from yorishiro.project import Project, find_project


FINALIZATION_SYSTEM_PROMPT = """\
You are synthesizing a complete, production-ready character soul document (SOUL.md) for use \
in AI roleplay simulation. This document must be detailed enough that an AI can accurately \
simulate this character's behavior across different contexts.

## Input

You will receive for one character:
1. Accumulated cross-scene insights
2. All per-scene extraction notes (CharacterSceneNote records as JSON)

## Output Requirements

1. **Every behavioral claim must have evidence**: Include at least one scene reference (chapter number).
2. **Include verbatim dialogue**: Preserve exact original-language quotes, never paraphrase.
3. **Write in the source language**: All content in Japanese/Chinese/etc. Section headings stay in English.
4. **Do not invent facts**: Only use information from the provided material. Mark uncertain areas.
5. **Default timeline**: Use the character's state at the END of the story as the default for simulation.

## Output Format

Produce a complete SOUL.md following this structure exactly:

---

# {Character Name}

> One-sentence identity summary capturing core drive and current role.

## 1. Core Identity
Core driving motivation; current life stage/role; what fundamentally animates this character.

## 2. Personality Model

### 2.1 Core Values
List 3-5 core values with behavioral evidence from scenes.

### 2.2 Motivations & Desires
- **Surface desire:** What they consciously pursue
- **Deep desire:** What they secretly want
- **Ultimate desire:** Their core life goal

### 2.3 Core Fears & Avoidances
What they fear most and what behaviors they avoid, with scene evidence.

### 2.4 Character Traits
- **Public persona:** How they present to others
- **True self:** Who they really are inside
- **Internal contradictions:** Tensions between these

### 2.5 Cognitive Patterns
How they think, process information, make decisions.

## 3. Voice & Language

### 3.1 Overall Register
Typical speech style, formality level, dialect.

### 3.2 Catchphrases & Signature Expressions
Frequently used phrases with context.

### 3.3 Sentence Style Preferences
Sentence length, structure preferences, rhetorical patterns.

### 3.4 Humor Style
How they use humor: self-deprecation, sarcasm, wordplay, etc.

### 3.5 Language Under Emotional Intensity
How speech changes under stress, anger, joy, grief.

### 3.6 Dialogue Samples
3-5 key dialogue excerpts with scene context, showing different emotional states.

### 3.7 Interaction Patterns Table
Create a table showing typical input→response patterns:

| Input Type | Character Response | Source |
|------------|-------------------|--------|
| Being praised | ... | chXXX |
| Being questioned about X | ... | chXXX |
| Facing Y situation | ... | chXXX |

## 4. Relationships

For each significant relationship, provide:
- **Nature**: What kind of relationship
- **Interaction pattern**: How they act around each other
- **Key turning points**: How the relationship changed

## 5. Behavioral Patterns

### 5.1 Under Pressure / Conflict
How they behave when stressed or in conflict.

### 5.2 With Intimacy / Trust
How they behave with people they trust.

### 5.3 Facing Failure / Setbacks
How they handle failure and setbacks.

### 5.4 Moral Dilemmas
How they approach ethical decisions.

### 5.5 Workplace / Senior Behavior
How they act in work/school contexts, as senior or junior.

### 5.6 Habitual Behaviors & Rituals
Daily habits, coping mechanisms, routines.

## 6. Negative Constraints

CRITICAL for preventing out-of-character behavior. Structure as three tiers:

### Absolutely Never
Things this character would never do under any circumstances, with reason.

### In X Context Will Not
Things they won't do in specific situations, with context.

### Contradictions / Traps
Behaviors that seem contradictory but aren't (e.g., "says X but always does Y").

## 7. Character Arc

**Before writing this section**: Scan all `active_persona` values from the scene notes.
Identify major timeline phases (e.g., "high school era" vs "10 years later").
Group scenes by phase before writing the subsections below.
Mark which phase represents the END state (default for simulation).

### 7.1 Starting State
Who they are at the beginning.

### 7.2 Key Turning Points
Major events that changed them, with chapter references.

### 7.3 Ending State
Who they become by the end.

### 7.4 Stage-by-Stage Personality Differences

Identify major timeline phases from active_persona patterns (e.g., ch002-009 high school, ch010+研究所長).
Create a comparison table with at least 2 phases:

| Stage | Core Drive | Key Relationships | Language/Register | Behavioral Focus |
|-------|------------|-------------------|-------------------|------------------|
| Phase1 (chXXX-YYY) | ... | ... | ... | ... |
| Phase2 (chXXX-YYY) | ... | ... | ... | ... |

Include at least one scene reference per phase.
Mark which phase is the DEFAULT (end state) for simulation.

## 8. World Knowledge

### Known Facts
What they know for certain.

### Unknown to Character
What the reader knows but they don't.

### Mistaken Beliefs
Things they believe that are wrong, and when/if corrected.

## 9. Simulation Directives

### 9.1 Default Timeline
Which story phase to simulate BY DEFAULT — must be the END state / final timeline phase.
Specify: (1) the phase name, (2) key behavioral characteristics of this phase,
(3) how to switch to other phases if requested.

### 9.2 Timeline Switching Rules
How behavior differs across phases; triggers for switching.

### 9.3 Dialogue Style Rules
Specific rules for speech patterns, formality, catchphrases.

### 9.4 Prohibited Behaviors Checklist
What not to do when roleplaying this character.

### 9.5 Fallback Strategies
What to do when uncertain about the character's response.

---

Note: This document may be further refined in cross-source alignment passes.
"""


class SoulDocOutput(BaseModel):
    """Finalization pass output: a complete structured SOUL.md."""

    content: str = Field(
        description="Complete SOUL.md document in markdown format.",
    )


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


def load_all_notes(characters_dir: Path, character_name: str) -> list[dict]:
    """Load all chapter notes for a character."""
    all_notes: list[dict] = []
    char_dir = characters_dir / character_name
    
    if not char_dir.exists():
        return all_notes
    
    for notes_file in sorted(char_dir.glob("ch*.json")):
        try:
            all_notes.extend(json.loads(notes_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    
    all_notes.sort(key=lambda n: (n.get("chapter_index", 0), n.get("scene_index", 0)))
    return all_notes


def load_insights(characters_dir: Path, character_name: str) -> str:
    """Load accumulated insights for a character."""
    insights_path = characters_dir / character_name / "insights.md"
    if insights_path.exists():
        return insights_path.read_text(encoding="utf-8")
    return "(no insights accumulated yet)"


async def synthesize_character(
    character_name: str,
    characters_dir: Path,
    souls_dir: Path,
    agent,
) -> None:
    """Generate final SOUL.md for one character."""
    insights = load_insights(characters_dir, character_name)
    all_notes = load_all_notes(characters_dir, character_name)
    
    notes_json = json.dumps(all_notes, ensure_ascii=False, indent=2)
    prompt = (
        f"# Character: {character_name}\n\n"
        f"## Accumulated Insights\n\n{insights}\n\n"
        f"## All Scene Extraction Notes ({len(all_notes)} scenes)\n\n"
        + notes_json
    )
    
    print(
        f"  Prompt sizes: insights={len(insights)} chars, "
        f"notes={len(notes_json)} chars, total={len(prompt)} chars",
        file=sys.stderr,
    )
    
    result = await agent_run_with_retry(agent, prompt)
    soul_doc_content: str = result.output.content
    
    souls_dir.mkdir(parents=True, exist_ok=True)
    safe_name = character_name.replace("/", "_").replace("\\", "_").replace("\0", "")
    out_path = souls_dir / f"{safe_name}.md"
    out_path.write_text(soul_doc_content, encoding="utf-8")
    print(f"  Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize final SOUL.md documents from accumulated character data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.synthesize --project projects/CPK\n"
            "  uv run python -m yorishiro.synthesize --project projects/CPK --character 酒寄彩葉\n"
        ),
    )
    
    # Project mode arguments
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory (enables project mode)",
    )
    parser.add_argument(
        "--character",
        type=str,
        nargs="+",
        default=None,
        help="Character name(s) to synthesize (default: all characters)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source ID to process (default: first source in project)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing SOUL.md files")
    add_model_args(parser)
    args = parser.parse_args()

    # Resolve project
    project: Project | None = None
    if args.project:
        project = Project.load(args.project)
    else:
        project = find_project(Path.cwd())
    
    if not project:
        print("Error: No project found. Use --project or run from within a project directory.", file=sys.stderr)
        sys.exit(1)
    
    # Get source configuration
    source_id = args.source
    if not source_id:
        if project.sources:
            source_id = project.sources[0].id
        else:
            print("Error: No sources defined in project.", file=sys.stderr)
            sys.exit(1)
    
    # Resolve paths
    characters_dir = project.source_dir(source_id) / "characters"
    souls_dir = project.souls_dir()
    
    if not characters_dir.exists():
        print(f"Error: Characters directory not found: {characters_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Get model config
    config = project.resolved_model_config("synthesize")
    
    # Resolve characters to process
    if args.character:
        target_characters = args.character
    else:
        # Find all characters with notes
        target_characters = []
        for char_dir in characters_dir.iterdir():
            if char_dir.is_dir() and (char_dir / "insights.md").exists():
                target_characters.append(char_dir.name)
        
        if not target_characters:
            print(f"Error: No character data found in {characters_dir}", file=sys.stderr)
            sys.exit(1)
    
    # Filter out characters that already have SOUL.md unless --force
    if not args.force:
        pending = []
        for name in target_characters:
            safe_name = name.replace("/", "_").replace("\\", "_").replace("\0", "")
            soul_path = souls_dir / f"{safe_name}.md"
            if soul_path.exists():
                print(f"Skipping: {soul_path} already exists (use --force to overwrite)")
            else:
                pending.append(name)
        target_characters = pending
    
    if not target_characters:
        print("No characters to process.", file=sys.stderr)
        sys.exit(0)
    
    agent = build_agent_from_args(
        args,
        output_type=SoulDocOutput,
        system_prompt=FINALIZATION_SYSTEM_PROMPT,
        config=config,
    )
    
    model_display = args.model or (config.name if config else "unknown")
    print(f"Synthesizing {len(target_characters)} characters with {model_display} ...")
    
    async def run() -> None:
        for name in target_characters:
            print(f"\nSynthesizing 「{name}」 ...", file=sys.stderr)
            await synthesize_character(name, characters_dir, souls_dir, agent)
    
    asyncio.run(run())
    print("\nDone.")


if __name__ == "__main__":
    main()