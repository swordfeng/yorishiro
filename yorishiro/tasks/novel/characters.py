"""novel.characters step: extract per-scene character notes using an LLM agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from yorishiro.novel.character_extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    CharacterExtractionAgent,
    build_batches,
    load_aliases,
    load_all_scenes,
    make_extraction_models,
    process_all_batches,
)
from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class NovelCharactersTask(Task):
    """Extract character notes for all characters in a source."""

    def __init__(
        self,
        scenes_dir: Path,
        aliases_file: Path,
        output_dir: Path,
        project_config: Path,
        runtime: StepRuntime,
        batch_tokens: int = 32000,
        target_characters: list[str] | None = None,
    ) -> None:
        self._scenes_dir = scenes_dir
        self._aliases_file = aliases_file
        self._output_dir = output_dir
        self._project_config = project_config
        self._runtime = runtime
        self._batch_tokens = batch_tokens
        self._target_characters = target_characters

    def input_paths(self) -> list[Path]:
        return [self._aliases_file, self._project_config]

    def output_paths(self) -> list[Path]:
        # Return existing insights.md files as output markers; fall back to placeholder
        existing = sorted(self._output_dir.glob("*/insights.md"))
        return existing if existing else [self._output_dir / "__placeholder__"]

    def completion_marker(self) -> Path:
        # Use the most recently written insights.md across all characters
        existing = sorted(self._output_dir.glob("*/insights.md"))
        return existing[-1] if existing else self._output_dir / "__placeholder__"

    def _run(self) -> None:
        aliases = load_aliases(self._aliases_file)
        target_characters = self._target_characters or list(aliases.keys())

        print(f"[novel.characters] Target: {target_characters}")
        all_scenes = load_all_scenes(self._scenes_dir, aliases, target_characters)
        if not all_scenes:
            raise RuntimeError(
                f"No scenes found in {self._scenes_dir}. Run novel.scenes first."
            )

        batches = build_batches(all_scenes, self._batch_tokens)
        print(f"[novel.characters] {len(all_scenes)} scenes → {len(batches)} batches.")

        self._output_dir.mkdir(parents=True, exist_ok=True)

        async def run() -> None:
            _, BatchExtractionResult = make_extraction_models(target_characters)
            agent = self._runtime.agent(
                output_type=BatchExtractionResult,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
            )
            await process_all_batches(
                batches=batches,
                target_characters=target_characters,
                characters_dir=self._output_dir,
                output_dir=self._output_dir,
                agent=cast(CharacterExtractionAgent, agent),
            )

        asyncio.run(run())
        print("[novel.characters] Done.")


class NovelCharactersStep(Step):
    step_id = "novel.characters"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        runtime = self._registry.for_step(self.step_id)
        step_cfg = self._project.step_config(self.step_id)
        batch_tokens = step_cfg.get("batch_tokens", 32000)
        return [
            NovelCharactersTask(
                scenes_dir=self._project.step_dir(self._source_id, "scenes"),
                aliases_file=self._project.step_dir(self._source_id, "aliases") / "character_aliases.json",
                output_dir=self._project.step_dir(self._source_id, "characters"),
                project_config=self._project.config_path,
                runtime=runtime,
                batch_tokens=batch_tokens,
            )
        ]
