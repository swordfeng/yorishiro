"""Pipeline orchestrator — resolves step IDs to Step instances and runs them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from yorishiro.backup import ProjectBackup
from yorishiro.project import Project
from yorishiro.tasks.base import Step
from yorishiro.tasks.registry import ModelRegistry

if TYPE_CHECKING:
    pass

# Step IDs that apply to novel sources (in dependency order)
NOVEL_STEPS = [
    "novel.chapters",
    "novel.scenes",
    "novel.aliases",
    "novel.characters",
]

# Step IDs that apply to film sources (in dependency order)
FILM_STEPS = [
    "film.shots",
    "film.frames",
    "film.audio.extract",
    "film.audio.separate",
    "film.audio.vad",
    "film.audio.diarize",
    "film.audio.stt",
    "film.audio.emotion",
    "film.audio.analysis",
    "film.shot_groups",
    "film.scenes",
]

# Cross-source steps (in dependency order)
CROSS_STEPS = [
    "cross.synthesize",
]

# All steps in full execution order
ALL_STEPS = NOVEL_STEPS + FILM_STEPS + CROSS_STEPS

# Pseudo source_id used for cross-source steps
CROSS_SOURCE_ID = "__cross__"


def expand_step_prefix(prefix: str) -> list[str]:
    """Expand a step prefix to all matching leaf step IDs in canonical order.

    Examples:
        "film.audio" → ["film.audio.extract", "film.audio.separate", ...]
        "film"       → all FILM_STEPS
        "novel"      → all NOVEL_STEPS
        "film.shots" → ["film.shots"]  (exact match, returned as-is)
    """
    if prefix in ALL_STEPS:
        return [prefix]
    match = prefix.rstrip(".") + "."
    matched = [s for s in ALL_STEPS if s.startswith(match)]
    if not matched:
        raise ValueError(f"No steps match prefix: {prefix!r}")
    return matched


def _build_step(step_id: str, source_id: str, project: Project, registry: ModelRegistry) -> Step:
    """Construct the Step implementation for a given step_id."""
    # Novel steps
    if step_id == "novel.chapters":
        from yorishiro.tasks.novel.chapters import NovelChaptersStep
        return NovelChaptersStep(project, source_id, registry)
    if step_id == "novel.scenes":
        from yorishiro.tasks.novel.scenes import NovelScenesStep
        return NovelScenesStep(project, source_id, registry)
    if step_id == "novel.aliases":
        from yorishiro.tasks.novel.aliases import NovelAliasesStep
        return NovelAliasesStep(project, source_id, registry)
    if step_id == "novel.characters":
        from yorishiro.tasks.novel.characters import NovelCharactersStep
        return NovelCharactersStep(project, source_id, registry)

    # Film steps
    if step_id == "film.shots":
        from yorishiro.tasks.film.shots import FilmShotsStep
        return FilmShotsStep(project, source_id, registry)
    if step_id == "film.frames":
        from yorishiro.tasks.film.frames import FilmFramesStep
        return FilmFramesStep(project, source_id, registry)
    if step_id == "film.audio.extract":
        from yorishiro.tasks.film.audio import FilmAudioExtractStep
        return FilmAudioExtractStep(project, source_id, registry)
    if step_id == "film.audio.separate":
        from yorishiro.tasks.film.audio import FilmAudioSeparateStep
        return FilmAudioSeparateStep(project, source_id, registry)
    if step_id == "film.audio.vad":
        from yorishiro.tasks.film.audio import FilmAudioVADStep
        return FilmAudioVADStep(project, source_id, registry)
    if step_id == "film.audio.diarize":
        from yorishiro.tasks.film.audio import FilmAudioDiarizeStep
        return FilmAudioDiarizeStep(project, source_id, registry)
    if step_id == "film.audio.stt":
        from yorishiro.tasks.film.audio import FilmAudioSTTStep
        return FilmAudioSTTStep(project, source_id, registry)
    if step_id == "film.audio.emotion":
        from yorishiro.tasks.film.audio import FilmAudioEmotionStep
        return FilmAudioEmotionStep(project, source_id, registry)
    if step_id == "film.audio.analysis":
        from yorishiro.tasks.film.audio import FilmAudioAnalysisStep
        return FilmAudioAnalysisStep(project, source_id, registry)
    if step_id == "film.shot_groups":
        from yorishiro.tasks.film.shot_groups import FilmShotGroupsStep
        return FilmShotGroupsStep(project, source_id, registry)
    if step_id == "film.scenes":
        from yorishiro.tasks.film.scenes import FilmScenesStep
        return FilmScenesStep(project, source_id, registry)

    # Cross steps
    if step_id == "cross.synthesize":
        from yorishiro.tasks.cross.synthesize import CrossSynthesizeStep
        return CrossSynthesizeStep(project, source_id, registry)

    raise ValueError(f"Unknown step_id: {step_id!r}")


class Orchestrator:
    """Resolve and run pipeline steps for a project."""

    def __init__(self, project: Project) -> None:
        self._project = project
        self._registry = ModelRegistry(project)
        self._backup = ProjectBackup(project.root)

    def run(
        self,
        step_ids: list[str],
        source_id: str,
        force: bool = False,
        task_key: str | None = None,
        backup_after: bool = True,
    ) -> None:
        """Run the given steps for a specific source.

        Each entry in step_ids may be a leaf step ID or a prefix that expands
        to multiple leaf steps (e.g. "film.audio" or "film").
        """
        leaf_ids: list[str] = []
        for sid in step_ids:
            leaf_ids.extend(expand_step_prefix(sid))
        for step_id in leaf_ids:
            step = _build_step(step_id, source_id, self._project, self._registry)
            step.run(force=force, task_key=task_key)
        if backup_after:
            label = f"{'+'.join(step_ids)}-{source_id}"
            self._backup.snapshot(label)

    def run_group(self, group_name: str, source_id: str, force: bool = False) -> None:
        """Run a named step group for a specific source."""
        steps = self._project.step_groups.get(group_name)
        if steps is None:
            raise ValueError(f"Unknown step group: {group_name!r}")
        self.run(steps, source_id, force=force)

    def run_all(self, force: bool = False) -> None:
        """Run all steps for all sources, then cross-source steps."""
        for source in self._project.sources:
            if source.type == "novel":
                self.run(NOVEL_STEPS, source.id, force=force, backup_after=False)
            elif source.type == "film":
                self.run(FILM_STEPS, source.id, force=force, backup_after=False)

        self.run(CROSS_STEPS, CROSS_SOURCE_ID, force=force, backup_after=False)
        self._backup.snapshot("all")

    def status(self) -> None:
        """Print completion status for all steps across all sources."""
        for source in self._project.sources:
            steps = NOVEL_STEPS if source.type == "novel" else FILM_STEPS
            for step_id in steps:
                step = _build_step(step_id, source.id, self._project, self._registry)
                complete = step.is_complete()
                mark = "✓" if complete else "✗"
                print(f"  [{mark}] {step_id:<24} {source.id}")

        for step_id in CROSS_STEPS:
            step = _build_step(step_id, CROSS_SOURCE_ID, self._project, self._registry)
            complete = step.is_complete()
            mark = "✓" if complete else "✗"
            print(f"  [{mark}] {step_id:<24} (cross)")
