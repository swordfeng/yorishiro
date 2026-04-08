"""novel.scenes step orchestration."""

from __future__ import annotations

import re
from pathlib import Path

from yorishiro.novel.scene_segmentation import SceneSegmentationConfig, process_chapter
from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class NovelScenesTask(Task):
    """Segment one chapter file into scenes."""

    def __init__(
        self,
        chapter_path: Path,
        output_dir: Path,
        project_yaml: Path,
        runtime: StepRuntime,
        segmentation_config: SceneSegmentationConfig,
    ) -> None:
        self.key = chapter_path.stem
        self._chapter_path = chapter_path
        self._output_dir = output_dir
        self._project_yaml = project_yaml
        self._runtime = runtime
        self._segmentation_config = segmentation_config

    def input_paths(self) -> list[Path]:
        return [self._chapter_path, self._project_yaml]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "scenes_manifest.json"]

    def _run(self) -> None:
        print(f"[novel.scenes] Processing {self._chapter_path.name} ...")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        process_chapter(
            chapter_file=self._chapter_path,
            output_dir=self._output_dir,
            force=True,
            runtime=self._runtime,
            segmentation_config=self._segmentation_config,
            material_yaml=self._project_yaml,
        )


class NovelScenesStep(Step):
    step_id = "novel.scenes"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        runtime = self._registry.for_step(self.step_id)
        step_config = self._project.step_config(self.step_id)
        segmentation_config = SceneSegmentationConfig.from_step_config(step_config)
        chapters = self._project.list_chapters(self._source_id)
        scenes_dir = self._project.step_dir(self._source_id, "scenes")
        project_yaml = self._project.root / "project.yaml"

        return [
            NovelScenesTask(
                chapter_path=chapter_path,
                output_dir=scenes_dir / chapter_path.stem,
                project_yaml=project_yaml,
                runtime=runtime,
                segmentation_config=segmentation_config,
            )
            for chapter_path in chapters
        ]

    def _chapter_stem_to_index(self, stem: str) -> int:
        m = re.search(r"\d+", stem)
        return int(m.group()) if m else 0
