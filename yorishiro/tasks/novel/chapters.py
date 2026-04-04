"""novel.chapters step: extract chapters from EPUB source."""

from __future__ import annotations

from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry


class NovelChaptersTask(Task):
    """Extract all chapters from an EPUB file into ch*.txt files."""

    def __init__(self, epub_path: Path, output_dir: Path) -> None:
        self._epub_path = epub_path
        self._output_dir = output_dir

    def input_paths(self) -> list[Path]:
        return [self._epub_path]

    def output_paths(self) -> list[Path]:
        existing = sorted(self._output_dir.glob("ch*.txt"))
        return existing if existing else [self._output_dir / "ch000.txt"]

    def completion_marker(self) -> Path:
        existing = sorted(self._output_dir.glob("ch*.txt"))
        if existing:
            return existing[-1]  # last chapter written last
        return self._output_dir / "ch000.txt"

    def _run(self) -> None:
        from yorishiro.chapters_epub import extract_chapters

        print(f"[novel.chapters] Extracting chapters from {self._epub_path.name} ...")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved = extract_chapters(self._epub_path, self._output_dir)
        print(f"[novel.chapters] Wrote {len(saved)} chapter files.")


class NovelChaptersStep(Step):
    step_id = "novel.chapters"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id

    def tasks(self) -> list[Task]:
        epub_path = self._project.get_source_path(self._source_id)
        output_dir = self._project.step_dir(self._source_id, "chapters")
        return [NovelChaptersTask(epub_path, output_dir)]
