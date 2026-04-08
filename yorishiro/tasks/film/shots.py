"""film.shots step: detect shot boundaries in a video file."""

from __future__ import annotations

from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class FilmShotsTask(Task):
    """Detect shot boundaries in a video file, writing shots.json."""

    def __init__(self, video_path: Path, output_dir: Path, runtime: StepRuntime) -> None:
        self._video_path = video_path
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [self._video_path]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "shots.json"]

    def _run(self) -> None:
        detector = self._runtime.instance()
        print(f"[film.shots] Detecting shots in {self._video_path.name} ...")
        shot_list = detector.detect(self._video_path, self._output_dir, force=True)
        print(f"[film.shots] Detected {len(shot_list.shots)} shots.")


class FilmShotsStep(Step):
    step_id = "film.shots"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        video_path = self._project.get_source_path(self._source_id)
        output_dir = self._project.step_dir(self._source_id, "shots")
        return [FilmShotsTask(video_path, output_dir, self._registry.for_step(self.step_id))]
