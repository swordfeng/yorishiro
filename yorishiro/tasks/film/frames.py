"""film.frames step: extract keyframes from detected shots."""

from __future__ import annotations

import json
from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class FilmFramesTask(Task):
    """Extract keyframes from video shots, writing frame_index.json."""

    def __init__(
        self,
        video_path: Path,
        shots_json: Path,
        output_dir: Path,
        runtime: StepRuntime,
    ) -> None:
        self._video_path = video_path
        self._shots_json = shots_json
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [self._video_path, self._shots_json]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "frame_index.json"]

    def _run(self) -> None:
        from yorishiro.models.film_models import ShotList

        if not self._shots_json.exists():
            raise FileNotFoundError(f"shots.json not found: {self._shots_json}")

        shot_list = ShotList(**json.loads(self._shots_json.read_text(encoding="utf-8")))
        extractor = self._runtime.instance()

        print(f"[film.frames] Extracting keyframes for {len(shot_list.shots)} shots ...")
        keyframes = extractor.extract(self._video_path, shot_list, self._output_dir, force=True)
        print(f"[film.frames] Done. {len(keyframes)} shot keyframe sets.")


class FilmFramesStep(Step):
    step_id = "film.frames"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        video_path = self._project.get_source_path(self._source_id)
        shots_dir = self._project.step_dir(self._source_id, "shots")
        frames_dir = self._project.step_dir(self._source_id, "frames")
        return [FilmFramesTask(
            video_path,
            shots_dir / "shots.json",
            frames_dir,
            self._registry.for_step(self.step_id),
        )]
