"""Base abstractions for Yorishiro pipeline tasks and steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Task(ABC):
    """Atomic unit of work with defined input/output paths and staleness logic.

    Subclasses must implement:
      - output_paths() — files produced by this task
      - input_paths()  — files consumed by this task
      - _run()         — the actual computation (called unconditionally)

    Optionally override:
      - completion_marker() — the single file used for staleness check
        (defaults to first entry of output_paths())
      - key — short string identifier for selecting this task within a Step
    """

    key: str | None = None

    @abstractmethod
    def output_paths(self) -> list[Path]:
        """Files produced by this task."""
        ...

    @abstractmethod
    def input_paths(self) -> list[Path]:
        """Files consumed by this task. If any is newer than the marker, task is stale."""
        ...

    def completion_marker(self) -> Path:
        """Single file whose mtime represents task completion.

        Default: first entry of output_paths(). Override when the natural
        completion marker is not the first output (e.g. an index file written
        after many individual scene files).
        """
        outputs = self.output_paths()
        if not outputs:
            raise ValueError(f"{self!r}: output_paths() returned empty list")
        return outputs[0]

    def is_stale(self) -> bool:
        """True if the task needs to run: marker missing or older than any input."""
        marker = self.completion_marker()
        if not marker.exists():
            return True
        t = marker.stat().st_mtime
        return any(
            src.exists() and src.stat().st_mtime > t for src in self.input_paths()
        )

    def run(self, force: bool = False) -> bool:
        """Execute this task if stale (or if force=True).

        Returns True if the task executed, False if skipped.
        """
        if not force and not self.is_stale():
            label = (
                f"{self.__class__.__name__}[{self.key}]"
                if self.key
                else self.__class__.__name__
            )
            print(f"  [skip] {label} — up to date")
            return False
        self._run()
        return True

    @abstractmethod
    def _run(self) -> None:
        """Unconditionally execute the task. Called by run()."""
        ...

    def __repr__(self) -> str:
        key_part = f"[{self.key}]" if self.key else ""
        return f"{self.__class__.__name__}{key_part}"


class Step(ABC):
    """Named pipeline step that owns one or more Tasks.

    Simple steps (e.g. shot detection) have a single Task.
    Batched steps (e.g. scene segmentation) have one Task per unit of work
    (one per chapter, one per character, etc.).
    """

    step_id: str

    @abstractmethod
    def tasks(self) -> list[Task]:
        """Return all tasks in execution order."""
        ...

    def run(self, force: bool = False, task_key: str | None = None) -> bool:
        """Run all tasks, or just the one matching task_key.

        Returns True if any task actually executed, False if all were skipped.

        Args:
            force: Re-run even if the task is not stale.
            task_key: If given, run only the task whose .key matches.
        """
        all_tasks = self.tasks()
        if task_key is not None:
            all_tasks = [t for t in all_tasks if t.key == task_key]
            if not all_tasks:
                raise ValueError(
                    f"No task with key={task_key!r} in step {self.step_id!r}"
                )
        ran = False
        for task in all_tasks:
            if task.run(force=force):
                ran = True
        return ran

    def is_complete(self) -> bool:
        """True when all tasks are up to date."""
        return all(not t.is_stale() for t in self.tasks())
