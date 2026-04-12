"""novel.aliases step: resolve character name aliases from scene texts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from yorishiro.novel.alias_resolution import (
    AliasAgent,
    BATCH_SYSTEM_PROMPT,
    RETRY_SYSTEM_PROMPT,
    SEED_SYSTEM_PROMPT,
    BatchUpdateResult,
    GlobalState,
    MissedAliasResolution,
    SeedFromSoulDocsResult,
    build_batches,
    load_all_scenes,
    process_all_batches,
    seed_from_insight_drafts,
    write_character_aliases,
    write_insight_drafts,
)
from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class NovelAliasesTask(Task):
    """Resolve character aliases from all scenes in a source."""

    def __init__(
        self,
        scenes_dir: Path,
        output_dir: Path,
        project_config: Path,
        runtime: StepRuntime,
        batch_tokens: int = 32000,
    ) -> None:
        self._scenes_dir = scenes_dir
        self._output_dir = output_dir
        self._project_config = project_config
        self._runtime = runtime
        self._batch_tokens = batch_tokens

    def input_paths(self) -> list[Path]:
        manifests = sorted(self._scenes_dir.rglob("scenes_manifest.json"))
        return manifests

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "character_aliases.json"]

    def _run(self) -> None:
        print(f"[novel.aliases] Loading scenes from {self._scenes_dir} ...")
        all_scenes = load_all_scenes(self._scenes_dir)
        if not all_scenes:
            raise RuntimeError(
                f"No scenes found in {self._scenes_dir}. Run novel.scenes first."
            )
        chapter_count = len({s.chapter for s in all_scenes})
        print(f"  Loaded {len(all_scenes)} scenes from {chapter_count} chapters.")

        batches = build_batches(all_scenes, self._batch_tokens)
        print(f"  Split into {len(batches)} batches.")

        async def run() -> GlobalState:
            batch_agent = self._runtime.agent(
                output_type=BatchUpdateResult,
                system_prompt=BATCH_SYSTEM_PROMPT,
            )
            retry_agent = self._runtime.agent(
                output_type=MissedAliasResolution,
                system_prompt=RETRY_SYSTEM_PROMPT,
            )
            initial_state = GlobalState()
            if self._output_dir.exists():
                seed_agent = self._runtime.agent(
                    output_type=SeedFromSoulDocsResult,
                    system_prompt=SEED_SYSTEM_PROMPT,
                )
                initial_state = await seed_from_insight_drafts(self._output_dir, cast(AliasAgent, seed_agent))
            return await process_all_batches(
                batches,
                initial_state,
                cast(AliasAgent, batch_agent),
                cast(AliasAgent, retry_agent),
            )

        print("[novel.aliases] Processing ...")
        state = asyncio.run(run())

        self._output_dir.mkdir(parents=True, exist_ok=True)
        aliases_path = self._output_dir / "character_aliases.json"
        write_character_aliases(state, aliases_path)
        write_insight_drafts(state, self._output_dir)
        print(f"[novel.aliases] Done. {len(state.characters)} characters resolved.")


class NovelAliasesStep(Step):
    step_id = "novel.aliases"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        runtime = self._registry.for_step(self.step_id)
        step_cfg = self._project.step_config(self.step_id)
        batch_tokens = step_cfg.get("batch_tokens", 32000)
        return [
            NovelAliasesTask(
                scenes_dir=self._project.step_dir(self._source_id, "scenes"),
                output_dir=self._project.step_dir(self._source_id, "aliases"),
                project_config=self._project.config_path,
                runtime=runtime,
                batch_tokens=batch_tokens,
            )
        ]
