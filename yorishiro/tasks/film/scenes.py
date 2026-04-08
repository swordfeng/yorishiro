"""film.scenes step: analyze each scene group using a VLM agent."""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class FilmScenesTask(Task):
    """Run VLM scene analysis for all shot groups, writing scene_index.json."""

    def __init__(
        self,
        shot_groups_json: Path,
        shots_json: Path,
        frames_dir: Path,
        audio_dir: Path,
        output_dir: Path,
        project_name: str,
        runtime: StepRuntime,
        max_frames_per_scene: int = 8,
    ) -> None:
        self._shot_groups_json = shot_groups_json
        self._shots_json = shots_json
        self._frames_dir = frames_dir
        self._audio_dir = audio_dir
        self._output_dir = output_dir
        self._project_name = project_name
        self._runtime = runtime
        self._max_frames_per_scene = max_frames_per_scene

    def input_paths(self) -> list[Path]:
        return [self._shot_groups_json, self._shots_json, self._audio_dir / "transcript.json"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "scene_index.json"]

    def completion_marker(self) -> Path:
        return self._output_dir / "scene_index.json"

    def _run(self) -> None:
        from yorishiro.agent_utils import estimate_tokens
        from yorishiro.agents.film.scene_analysis import SceneAnalysisAgent
        from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
        from yorishiro.models.film_models import (
            Frame,
            FilmSceneIndex,
            FilmSceneContent,
            FilmSceneMetadata,
            KeyFrameSet,
            MusicSegment,
            ShotGroup,
            ShotList,
            SoundEvent,
            Transcript,
        )
        from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig

        shot_list = ShotList(**json.loads(self._shots_json.read_text(encoding="utf-8")))

        frame_index = json.loads((self._frames_dir / "frame_index.json").read_text(encoding="utf-8"))
        keyframes = [KeyFrameSet(**kf) for kf in frame_index.get("keyframes", [])]

        transcript = Transcript(**json.loads(
            (self._audio_dir / "transcript.json").read_text(encoding="utf-8")
        ))

        sound_events: list[SoundEvent] = []
        se_path = self._audio_dir / "sound_events.json"
        if se_path.exists():
            se_data = json.loads(se_path.read_text(encoding="utf-8"))
            sound_events = [SoundEvent(**e) for e in se_data.get("events", [])]

        music_segments: list[MusicSegment] = []
        m_path = self._audio_dir / "music_analysis.json"
        if m_path.exists():
            m_data = json.loads(m_path.read_text(encoding="utf-8"))
            music_segments = [MusicSegment(**m) for m in m_data.get("segments", [])]

        sg_data = json.loads(self._shot_groups_json.read_text(encoding="utf-8"))
        shot_groups = [ShotGroup(**g) for g in sg_data.get("shot_groups", [])]

        speaker_bank = SpeakerBankManager(SpeakerBankManagerConfig())
        speaker_bank.load(self._audio_dir)

        kf_extractor = KeyFrameExtractor(KeyFrameExtractorConfig(
            max_frames_per_scene=self._max_frames_per_scene,
        ))
        frame_base_path = self._frames_dir

        analysis_agent = SceneAnalysisAgent(
            self._runtime.agent(
                output_type=FilmSceneContent,
                system_prompt=SceneAnalysisAgent.SYSTEM_PROMPT,
            )
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        async def run() -> list[FilmSceneMetadata]:
            scenes_metadata: list[FilmSceneMetadata] = []

            for scene_num, group in enumerate(shot_groups, 1):
                scene_id = f"fs{scene_num:03d}"
                print(f"  [film.scenes] Analyzing {scene_id} ({len(group.shots)} shots) ...")

                first_shot = shot_list.get_shot(group.shots[0])
                last_shot = shot_list.get_shot(group.shots[-1])
                if first_shot is None or last_shot is None:
                    continue

                first_idx = int(group.shots[0].replace("sh", "")) - 1
                last_idx = int(group.shots[-1].replace("sh", "")) - 1
                keyframes_for_scene = keyframes[first_idx:last_idx + 1]

                all_frames: list[Frame] = []
                for kf in keyframes_for_scene:
                    all_frames.extend(kf.extracted_frames)

                selected_frames = kf_extractor.select_keyframes_for_scene(
                    all_frames, max_frames=self._max_frames_per_scene, base_path=frame_base_path
                )

                prompt = analysis_agent.build_analysis_prompt(
                    scene_group=group,
                    shot_list=shot_list,
                    selected_frames=selected_frames,
                    transcript=transcript,
                    sound_events=sound_events,
                    music=music_segments,
                    speaker_bank_manager=speaker_bank,
                    project_name=self._project_name,
                    scene_id=scene_id,
                    frame_base_path=frame_base_path,
                )

                try:
                    content = await analysis_agent.run(prompt)

                    for spk_id, name in content.speaker_map_update.items():
                        if speaker_bank.speaker_bank.get_speaker(spk_id) is not None:
                            speaker_bank.confirm_speaker(spk_id, name, scene_id)

                    scene_text = analysis_agent.format_scene_file(
                        scene_id=scene_id,
                        source_file=shot_list.video_file,
                        scene_start=first_shot.start_time,
                        scene_end=last_shot.end_time,
                        content=content,
                    )
                    scene_path = self._output_dir / f"scene_{scene_id}.txt"
                    scene_path.write_text(scene_text, encoding="utf-8")

                    scenes_metadata.append(FilmSceneMetadata(
                        scene_id=scene_id,
                        source={
                            "type": "film",
                            "file": shot_list.video_file,
                            "shots": group.shots,
                            "start_time": first_shot.start_time,
                            "end_time": last_shot.end_time,
                        },
                        content_file=str(scene_path.relative_to(self._output_dir.parent)),
                        token_estimate=estimate_tokens(scene_text),
                        location=content.location,
                        time_of_day=content.time_of_day,
                        characters={
                            "present": content.characters_present,
                            "mentioned": content.characters_mentioned,
                        },
                        audio={
                            "has_music": bool(music_segments),
                            "music_type": music_segments[0].music_type if music_segments else None,
                            "notable_sounds": [e.description for e in sound_events[:5]],
                        },
                        speaker_map_snapshot=speaker_bank.speaker_bank.speaker_map.copy(),
                    ))

                except Exception as e:
                    print(f"  [film.scenes] Error analyzing {scene_id}: {e}")

            speaker_bank.save(self._audio_dir)
            return scenes_metadata

        print(f"[film.scenes] Analyzing {len(shot_groups)} scenes ...")
        scenes_metadata = asyncio.run(run())

        index = FilmSceneIndex(
            metadata={
                "version": "1.0",
                "generated_at": datetime.datetime.now().isoformat(),
                "project": self._project_name,
                "source_file": shot_list.video_file,
                "source_type": "film",
                "total_scenes": len(scenes_metadata),
            },
            scenes=scenes_metadata,
        )
        index.save(self._output_dir / "scene_index.json")
        print(f"[film.scenes] Done. {len(scenes_metadata)} scenes written.")


class FilmScenesStep(Step):
    step_id = "film.scenes"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        runtime = self._registry.for_step(self.step_id)
        step_cfg = self._project.step_config(self.step_id)
        max_frames = step_cfg.get("max_frames_per_scene", 8)

        return [
            FilmScenesTask(
                shot_groups_json=self._project.step_dir(self._source_id, "shot_groups") / "shot_groups.json",
                shots_json=self._project.step_dir(self._source_id, "shots") / "shots.json",
                frames_dir=self._project.step_dir(self._source_id, "frames"),
                audio_dir=self._project.step_dir(self._source_id, "audio"),
                output_dir=self._project.step_dir(self._source_id, "scenes"),
                project_name=self._project.name,
                runtime=runtime,
                max_frames_per_scene=max_frames,
            )
        ]
