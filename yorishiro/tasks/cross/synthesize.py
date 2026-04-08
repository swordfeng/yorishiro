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
        souls_dir: Path,
        runtime: StepRuntime,
    ) -> None:
        self._characters_dirs = characters_dirs
        self._souls_dir = souls_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        paths: list[Path] = []
        for chars_dir in self._characters_dirs:
            paths.extend(chars_dir.glob("*/insights.md"))
            paths.extend(chars_dir.glob("*/ch*.json"))
        return paths

    def output_paths(self) -> list[Path]:
        existing = sorted(self._souls_dir.glob("*.md"))
        return existing if existing else [self._souls_dir / "__placeholder__"]

    def completion_marker(self) -> Path:
        existing = sorted(self._souls_dir.glob("*.md"))
        return existing[-1] if existing else self._souls_dir / "__placeholder__"

    def _run(self) -> None:
        from yorishiro.synthesize import FINALIZATION_SYSTEM_PROMPT, SoulDocOutput, synthesize_character

        # Collect all characters across all source character dirs
        target_characters: set[str] = set()
        for chars_dir in self._characters_dirs:
            for char_dir in chars_dir.iterdir():
                if char_dir.is_dir() and (char_dir / "insights.md").exists():
                    target_characters.add(char_dir.name)

        if not target_characters:
            print("[cross.synthesize] No characters with insights found.")
            return

        print(f"[cross.synthesize] Synthesizing {len(target_characters)} characters ...")

        agent = self._runtime.agent(
            output_type=SoulDocOutput,
            system_prompt=FINALIZATION_SYSTEM_PROMPT,
        )

        self._souls_dir.mkdir(parents=True, exist_ok=True)

        async def run() -> None:
            for name in sorted(target_characters):
                print(f"  Synthesizing 「{name}」 ...")
                # Use the first characters_dir that has this character
                chars_dir = next(
                    (d for d in self._characters_dirs if (d / name / "insights.md").exists()),
                    self._characters_dirs[0],
                )
                await synthesize_character(name, chars_dir, self._souls_dir, agent)

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
        # Also check old-style paths for compatibility
        for source in self._project.sources:
            legacy = self._project.source_dir(source.id) / "characters"
            if legacy.exists() and legacy not in characters_dirs:
                characters_dirs.append(legacy)

        return [
            CrossSynthesizeTask(
                characters_dirs=characters_dirs,
                souls_dir=self._project.souls_dir(),
                runtime=self._registry.for_step(self.step_id),
            )
        ]
