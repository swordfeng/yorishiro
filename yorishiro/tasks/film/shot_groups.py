"""film.shot_groups step: group shots into narrative scenes using an LLM agent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class FilmShotGroupsTask(Task):
    """Run batched LLM shot grouping, writing shot_groups.json."""

    def __init__(
        self,
        shots_json: Path,
        frames_dir: Path,
        transcript_json: Path,
        sound_events_json: Path,
        music_json: Path,
        output_dir: Path,
        runtime: StepRuntime,
        batch_size: int = 15,
    ) -> None:
        self._shots_json = shots_json
        self._frames_dir = frames_dir
        self._transcript_json = transcript_json
        self._sound_events_json = sound_events_json
        self._music_json = music_json
        self._output_dir = output_dir
        self._runtime = runtime
        self._batch_size = batch_size

    def input_paths(self) -> list[Path]:
        return [
            self._shots_json,
            self._frames_dir / "frame_index.json",
            self._transcript_json,
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "shot_groups.json"]

    def _run(self) -> None:
        from yorishiro.agents.film.shot_grouping import ShotGroupingAgent, build_shot_audio_summary
        from yorishiro.models.film_models import (
            GroupingBatchResult,
            KeyFrameSet,
            MusicSegment,
            ShotGroup,
            ShotList,
            SoundEvent,
            Transcript,
        )

        shot_list = ShotList(**json.loads(self._shots_json.read_text(encoding="utf-8")))

        frame_index = json.loads((self._frames_dir / "frame_index.json").read_text(encoding="utf-8"))
        keyframes = [KeyFrameSet(**kf) for kf in frame_index.get("keyframes", [])]

        transcript = Transcript(**json.loads(self._transcript_json.read_text(encoding="utf-8")))

        sound_events: list[SoundEvent] = []
        if self._sound_events_json.exists():
            se_data = json.loads(self._sound_events_json.read_text(encoding="utf-8"))
            sound_events = [SoundEvent(**e) for e in se_data.get("events", [])]

        music_segments: list[MusicSegment] = []
        if self._music_json.exists():
            m_data = json.loads(self._music_json.read_text(encoding="utf-8"))
            music_segments = [MusicSegment(**m) for m in m_data.get("segments", [])]

        grouping_agent = ShotGroupingAgent(
            self._runtime.agent(
                output_type=GroupingBatchResult,
                system_prompt=ShotGroupingAgent.SYSTEM_PROMPT,
            )
        )
        shots = shot_list.shots
        frame_base_path = self._frames_dir  # frame_path is relative to frames_dir

        async def run() -> list[ShotGroup]:
            all_groups: list[ShotGroup] = []
            previous_summary = ""
            partial_group: ShotGroup | None = None
            idx = 0

            while idx < len(shots):
                batch_end = min(idx + self._batch_size, len(shots))
                batch_shots = shots[idx:batch_end]
                batch_keyframes = keyframes[idx:batch_end]

                print(f"  [film.shot_groups] Batch {idx // self._batch_size + 1}: shots {idx + 1}-{batch_end}")

                audio_summaries = [
                    build_shot_audio_summary(shot, transcript.entries, sound_events, music_segments)
                    for shot in batch_shots
                ]

                prompt = grouping_agent.build_batch_prompt(
                    previous_summary=previous_summary,
                    partial_group=partial_group,
                    shots=batch_shots,
                    keyframes=batch_keyframes,
                    audio_summary="\n---\n".join(audio_summaries),
                    batch_num=idx // self._batch_size + 1,
                    total_batches=(len(shots) + self._batch_size - 1) // self._batch_size,
                    frame_base_path=frame_base_path,
                )

                try:
                    result = await grouping_agent.run(prompt)
                    for group in result.new_scene_groups:
                        if group.is_complete:
                            all_groups.append(group)
                    if result.new_scene_groups:
                        last = result.new_scene_groups[-1]
                        partial_group = None if last.is_complete else last
                    previous_summary = result.summary_update
                    idx = max(result.next_shot_index, batch_end)
                except Exception as e:
                    print(f"  [film.shot_groups] Error in batch: {e}")
                    idx = batch_end

            if partial_group:
                all_groups.append(partial_group)
            return all_groups

        print(f"[film.shot_groups] Grouping {len(shots)} shots ...")
        shot_groups = asyncio.run(run())
        print(f"[film.shot_groups] {len(shots)} shots → {len(shot_groups)} scenes.")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out = {"shot_groups": [g.model_dump() for g in shot_groups]}
        (self._output_dir / "shot_groups.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class FilmShotGroupsStep(Step):
    step_id = "film.shot_groups"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        runtime = self._registry.for_step(self.step_id)
        step_cfg = self._project.step_config(self.step_id)
        batch_size = step_cfg.get("batch_size", 15)

        shots_dir = self._project.step_dir(self._source_id, "shots")
        frames_dir = self._project.step_dir(self._source_id, "frames")
        audio_dir = self._project.step_dir(self._source_id, "audio")
        output_dir = self._project.step_dir(self._source_id, "shot_groups")

        return [
            FilmShotGroupsTask(
                shots_json=shots_dir / "shots.json",
                frames_dir=frames_dir,
                transcript_json=audio_dir / "transcript.json",
                sound_events_json=audio_dir / "sound_events.json",
                music_json=audio_dir / "music_analysis.json",
                output_dir=output_dir,
                runtime=runtime,
                batch_size=batch_size,
            )
        ]
