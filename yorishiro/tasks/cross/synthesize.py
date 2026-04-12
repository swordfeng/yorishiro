"""cross.synthesize step: generate final SOUL.md for each character."""

from __future__ import annotations

import asyncio
from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class CrossSynthesizeTask(Task):
    """Synthesize SOUL.md for all characters from all source character notes."""

    def __init__(
        self,
        characters_dirs: list[Path],
        aliases_dirs: list[Path],
        souls_dir: Path,
        runtime: StepRuntime,
    ) -> None:
        self._characters_dirs = characters_dirs
        self._aliases_dirs = aliases_dirs
        self._souls_dir = souls_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        paths: list[Path] = []
        for chars_dir in self._characters_dirs:
            paths.extend(chars_dir.glob("*/insights.md"))
            paths.extend(chars_dir.glob("*/*.json"))
        for aliases_dir in self._aliases_dirs:
            paths.extend(aliases_dir.glob("*/insights.md"))
        return paths

    def output_paths(self) -> list[Path]:
        existing = sorted(self._souls_dir.glob("*.md"))
        return existing if existing else [self._souls_dir / "__placeholder__"]

    def completion_marker(self) -> Path:
        existing = sorted(self._souls_dir.glob("*.md"))
        return existing[-1] if existing else self._souls_dir / "__placeholder__"

    def _run(self) -> None:
        from yorishiro.synthesize import (
            FINALIZATION_SYSTEM_PROMPT,
            SoulDocOutput,
            discover_target_characters,
            synthesize_character,
        )

        target_characters = discover_target_characters(
            self._characters_dirs,
            self._aliases_dirs,
        )

        if not target_characters:
            print("[cross.synthesize] No character synthesis inputs found.")
            return

        print(f"[cross.synthesize] Synthesizing {len(target_characters)} characters ...")

        agent = self._runtime.agent(
            output_type=SoulDocOutput,
            system_prompt=FINALIZATION_SYSTEM_PROMPT,
        )

        self._souls_dir.mkdir(parents=True, exist_ok=True)

        async def run() -> None:
            for name in target_characters:
                print(f"  Synthesizing 「{name}」 ...")
                await synthesize_character(
                    name,
                    character_dirs=self._characters_dirs,
                    alias_dirs=self._aliases_dirs,
                    souls_dir=self._souls_dir,
                    agent=agent,
                )

        asyncio.run(run())
        print("[cross.synthesize] Done.")


class CrossSynthesizeStep(Step):
    step_id = "cross.synthesize"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._registry = registry

    def tasks(self) -> list[Task]:
        # Collect characters dirs from all novel sources
        characters_dirs = [
            self._project.step_dir(source.id, "characters")
            for source in self._project.sources
            if source.type == "novel"
        ]
        aliases_dirs = [
            self._project.step_dir(source.id, "aliases")
            for source in self._project.sources
            if source.type == "novel"
        ]
        # Also check old-style paths for compatibility
        for source in self._project.sources:
            legacy = self._project.source_dir(source.id) / "characters"
            if legacy.exists() and legacy not in characters_dirs:
                characters_dirs.append(legacy)
            legacy_aliases = self._project.source_dir(source.id) / "aliases"
            if legacy_aliases.exists() and legacy_aliases not in aliases_dirs:
                aliases_dirs.append(legacy_aliases)

        return [
            CrossSynthesizeTask(
                characters_dirs=characters_dirs,
                aliases_dirs=aliases_dirs,
                souls_dir=self._project.souls_dir(),
                runtime=self._registry.for_step(self.step_id),
            )
        ]
