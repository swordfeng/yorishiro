"""novel.chapters step orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yorishiro.novel.chapter_extraction import extract_chapters
from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry


class NovelChaptersTask(Task):
    """Extract all chapters from a source file into ch*.txt files."""

    def __init__(self, source_path: Path, output_dir: Path, source_config: dict[str, Any] | None = None) -> None:
        self._source_path = source_path
        self._output_dir = output_dir
        self._source_config = source_config or {}

    def input_paths(self) -> list[Path]:
        return [self._source_path]

    def output_paths(self) -> list[Path]:
        existing = sorted(self._output_dir.glob("ch*.txt"))
        return existing if existing else [self._output_dir / "ch000.txt"]

    def completion_marker(self) -> Path:
        existing = sorted(self._output_dir.glob("ch*.txt"))
        if existing:
            return existing[-1]
        return self._output_dir / "ch000.txt"

    def _run(self) -> None:
        print(f"[novel.chapters] Extracting chapters from {self._source_path.name} ...")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved = extract_chapters(self._source_path, self._output_dir, source_config=self._source_config)
        print(f"[novel.chapters] Wrote {len(saved)} chapter files.")


class NovelChaptersStep(Step):
    step_id = "novel.chapters"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id

    def tasks(self) -> list[Task]:
        source = self._project.get_source(self._source_id)
        if source is None:
            raise ValueError(f"Source '{self._source_id}' not found")
        source_path = self._project.get_source_path(self._source_id)
        output_dir = self._project.step_dir(self._source_id, "chapters")
        return [NovelChaptersTask(source_path, output_dir, source_config=source.config)]
