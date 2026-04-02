"""Main pipeline for film scene extraction.

Orchestrates the full pipeline:
1. Layer 0A: Video processing (shot detection, keyframe extraction)
2. Layer 0B: Audio processing (speech, sounds, music)
3. Step 1: Shot grouping (LLM-based scene segmentation)
4. Step 2: Scene analysis (VLM-based scene content generation)

Usage:
    uv run python -m yorishiro.pipelines.film_scene_extraction --project <dir> --source <id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yorishiro.project import Project
from yorishiro.video import ShotDetector, KeyFrameExtractor
from yorishiro.video.shot_detector import ShotDetectorConfig
from yorishiro.video.keyframe_extractor import KeyFrameExtractorConfig
from yorishiro.audio import SpeechPipeline, SpeakerBankManager, SoundEventDetector, MusicAnalyzer
from yorishiro.audio.speech_pipeline import SpeechPipelineConfig
from yorishiro.audio.speaker_bank import SpeakerBankManagerConfig
from yorishiro.audio.sound_event_detector import SoundEventDetectorConfig
from yorishiro.audio.music_analyzer import MusicAnalyzerConfig
from yorishiro.agent_utils import estimate_tokens
from yorishiro.agents.film.shot_grouping import ShotGroupingAgent, build_shot_audio_summary
from yorishiro.agents.film.scene_analysis import SceneAnalysisAgent
from yorishiro.models.film_models import (
    ShotList,
    KeyFrameSet,
    Frame,
    Transcript,
    SoundEvent,
    MusicSegment,
    ShotGroup,
    FilmSceneMetadata,
    FilmSceneIndex,
)


class FilmExtractionConfig(BaseModel):
    video: dict = Field(default_factory=lambda: {})
    audio: dict = Field(default_factory=lambda: {})
    grouping: dict = Field(default_factory=lambda: {})
    analysis: dict = Field(default_factory=lambda: {})

    batch_size: int = Field(default=15, description="Shots per grouping batch")


@dataclass
class FilmPipelineContext:
    project: Project
    source_config: Any
    output_dir: Path
    config: FilmExtractionConfig
    model_config: Any

    shot_detector: ShotDetector
    keyframe_extractor: KeyFrameExtractor
    speech_pipeline: SpeechPipeline
    speaker_bank: SpeakerBankManager
    sound_detector: SoundEventDetector
    music_analyzer: MusicAnalyzer

    shot_list: ShotList | None = None
    keyframes: list[KeyFrameSet] | None = None
    transcript: Transcript | None = None
    sound_events: list[SoundEvent] | None = None
    music_segments: list[MusicSegment] | None = None
    scene_index: FilmSceneIndex | None = None


class FilmSceneExtractionPipeline:
    """Main pipeline for extracting scenes from film sources."""

    def __init__(self, ctx: FilmPipelineContext):
        self.ctx = ctx

    @classmethod
    def create(
        cls,
        project: Project,
        source_id: str,
        config: FilmExtractionConfig | None = None,
        model_config: Any = None,
        force: bool = False,
    ) -> FilmSceneExtractionPipeline:
        """Create pipeline with default configurations."""
        source_config = project.get_source(source_id)
        if not source_config:
            raise ValueError(f"Source '{source_id}' not found in project")

        if source_config.type != "film":
            raise ValueError(f"Source '{source_id}' is not a film source (type: {source_config.type})")

        output_dir = project.source_dir(source_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = config or FilmExtractionConfig()
        model_cfg = model_config or project.resolved_model_config("scene")

        shot_detector = ShotDetector(ShotDetectorConfig(**cfg.video.get("shot_detection", {})))
        keyframe_extractor = KeyFrameExtractor(KeyFrameExtractorConfig(**cfg.video.get("keyframe", {})))
        speech_pipeline = SpeechPipeline(SpeechPipelineConfig(**cfg.audio.get("speech", {})))
        speaker_bank = SpeakerBankManager(SpeakerBankManagerConfig(**cfg.audio.get("speaker", {})))
        sound_detector = SoundEventDetector(SoundEventDetectorConfig(**cfg.audio.get("sound", {})))
        music_analyzer = MusicAnalyzer(MusicAnalyzerConfig(**cfg.audio.get("music", {})))

        ctx = FilmPipelineContext(
            project=project,
            source_config=source_config,
            output_dir=output_dir,
            config=cfg,
            model_config=model_cfg,
            shot_detector=shot_detector,
            keyframe_extractor=keyframe_extractor,
            speech_pipeline=speech_pipeline,
            speaker_bank=speaker_bank,
            sound_detector=sound_detector,
            music_analyzer=music_analyzer,
        )

        return cls(ctx)

    def run(self, force: bool = False) -> FilmSceneIndex:
        """Execute the full film extraction pipeline."""
        video_path = self.ctx.project.get_source_path(self.ctx.source_config.id)

        print(f"Processing film: {video_path.name}")
        print(f"Output directory: {self.ctx.output_dir}")

        self._run_layer_0a(video_path, force)
        self._run_layer_0b(video_path, force)
        self._run_step_1(force)
        self._run_step_2(force)

        if self.ctx.scene_index:
            self._write_index()
            return self.ctx.scene_index

        raise RuntimeError("Pipeline failed: no scene index generated")

    def _run_layer_0a(self, video_path: Path, force: bool) -> None:
        """Layer 0A: Video processing."""
        print("\n=== Layer 0A: Video Processing ===")

        cache_dir = self.ctx.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("Step 0A.1: Shot detection ...")
        self.ctx.shot_list = self.ctx.shot_detector.detect(video_path, cache_dir, force)

        print("Step 0A.2: Keyframe extraction ...")
        if self.ctx.shot_list:
            self.ctx.keyframes = self.ctx.keyframe_extractor.extract(
                video_path,
                self.ctx.shot_list,
                cache_dir,
                force,
            )

    def _run_layer_0b(self, video_path: Path, force: bool) -> None:
        """Layer 0B: Audio processing."""
        print("\n=== Layer 0B: Audio Processing ===")

        cache_dir = self.ctx.output_dir / "cache"
        language = self.ctx.source_config.config.get("language")

        print("Step 0B.1: Speech pipeline (VAD, diarization, STT) ...")
        self.ctx.transcript = self.ctx.speech_pipeline.process(
            video_path,
            cache_dir,
            language=language,
            force=force,
        )

        print("Step 0B.2: Speaker bank management — resolving local speaker IDs ...")
        self.ctx.speaker_bank.load(cache_dir)
        self._resolve_speaker_ids(cache_dir / "audio.wav", cache_dir)

        print("Step 0B.3: Sound event detection ...")
        self.ctx.sound_events = self.ctx.sound_detector.detect(
            cache_dir / "audio.wav",
            cache_dir,
            transcript_end=self.ctx.transcript.entries[-1].end if self.ctx.transcript.entries else 0.0,
            force=force,
        )

        print("Step 0B.4: Music analysis ...")
        self.ctx.music_segments = self.ctx.music_analyzer.analyze(
            video_path,
            cache_dir / "audio.wav",
            cache_dir,
            force=force,
        )

        self.ctx.speaker_bank.save(cache_dir)

    def _resolve_speaker_ids(self, audio_path: Path, cache_dir: Path) -> None:
        """Resolve diarization local speaker IDs (SPEAKER_XX) to global SPKR_XXX IDs."""
        if not self.ctx.transcript or not self.ctx.transcript.entries:
            return

        # Skip if already resolved (entries already have global SPKR_XXX format)
        if all(e.speaker_global.startswith("SPKR_") for e in self.ctx.transcript.entries):
            return

        # Collect unique local speakers and a representative segment for each
        local_speakers: dict[str, Any] = {}
        for entry in self.ctx.transcript.entries:
            if entry.speaker_global not in local_speakers:
                local_speakers[entry.speaker_global] = entry

        # Assign global IDs (one per unique local speaker)
        local_to_global: dict[str, str] = {}
        for local_id, rep_entry in local_speakers.items():
            if audio_path.exists():
                embedding = self.ctx.speaker_bank.extract_speaker_embedding(
                    audio_path, rep_entry.start, rep_entry.end
                )
            else:
                embedding = None
            global_id = self.ctx.speaker_bank.assign_global_speaker_id(
                local_id, embedding, rep_entry.start
            )
            local_to_global[local_id] = global_id
            print(f"    {local_id} → {global_id}")

        # Rewrite transcript entries with global IDs
        for entry in self.ctx.transcript.entries:
            entry.speaker_global = local_to_global.get(entry.speaker_global, entry.speaker_global)

        # Persist updated transcript
        cache_file = cache_dir / "transcript.json"
        cache_file.write_text(self.ctx.transcript.model_dump_json(indent=2), encoding="utf-8")

    def _run_step_1(self, force: bool) -> None:
        """Step 1: Shot grouping with LLM."""
        print("\n=== Step 1: Shot Grouping ===")

        if self.ctx.shot_list is None or self.ctx.keyframes is None:
            raise RuntimeError("Layer 0A must complete before Step 1")

        if self.ctx.transcript is None or self.ctx.sound_events is None or self.ctx.music_segments is None:
            raise RuntimeError("Layer 0B must complete before Step 1")

        shot_groups = asyncio.run(self._group_shots_batched())

        self._shot_groups = shot_groups
        print(f"Grouped {len(self.ctx.shot_list.shots)} shots into {len(shot_groups)} scenes")

    async def _group_shots_batched(self) -> list[ShotGroup]:
        """Progressive shot grouping."""
        args = argparse.Namespace(
            provider=self.ctx.model_config.provider,
            model=self.ctx.model_config.name,
            thinking=self.ctx.model_config.thinking,
            output_mode=self.ctx.model_config.output_mode,
        )

        grouping_agent = ShotGroupingAgent.create(args, self.ctx.model_config)

        shots = self.ctx.shot_list.shots if self.ctx.shot_list else []
        keyframes = self.ctx.keyframes or []
        batch_size = self.ctx.config.batch_size

        transcript_entries = self.ctx.transcript.entries if self.ctx.transcript else []
        sound_events = self.ctx.sound_events or []
        music_segments = self.ctx.music_segments or []

        all_groups: list[ShotGroup] = []
        previous_summary = ""
        partial_group: ShotGroup | None = None
        idx = 0

        cache_dir = self.ctx.output_dir / "cache"
        frame_base_path = cache_dir  # frame.frame_path is relative to cache_dir

        while idx < len(shots):
            batch_end = min(idx + batch_size, len(shots))
            batch_shots = shots[idx:batch_end]
            batch_keyframes = keyframes[idx:batch_end]

            print(f"  Processing batch {idx // batch_size + 1}: shots {idx + 1}-{batch_end}")

            audio_summaries = []
            for shot in batch_shots:
                summary = build_shot_audio_summary(
                    shot,
                    transcript_entries,
                    sound_events,
                    music_segments,
                )
                audio_summaries.append(summary)

            prompt = grouping_agent.build_batch_prompt(
                previous_summary=previous_summary,
                partial_group=partial_group,
                shots=batch_shots,
                keyframes=batch_keyframes,
                audio_summary="\n---\n".join(audio_summaries),
                batch_num=idx // batch_size + 1,
                total_batches=(len(shots) + batch_size - 1) // batch_size,
                frame_base_path=frame_base_path,
            )

            try:
                result = await grouping_agent.run(prompt)

                for group in result.new_scene_groups:
                    if group.is_complete:
                        all_groups.append(group)

                if result.new_scene_groups:
                    last_group = result.new_scene_groups[-1]
                    if not last_group.is_complete:
                        partial_group = last_group
                    else:
                        partial_group = None

                previous_summary = result.summary_update
                # Always advance at least to batch_end to prevent infinite loops
                idx = max(result.next_shot_index, batch_end)

            except Exception as e:
                print(f"  Error in grouping: {e}", file=sys.stderr)
                idx = batch_end

        if partial_group:
            all_groups.append(partial_group)

        return all_groups

    def _run_step_2(self, force: bool) -> None:
        """Step 2: Scene analysis with VLM."""
        print("\n=== Step 2: Scene Analysis ===")

        if not hasattr(self, "_shot_groups") or not self._shot_groups:
            raise RuntimeError("Step 1 must complete before Step 2")

        args = argparse.Namespace(
            provider=self.ctx.model_config.provider,
            model=self.ctx.model_config.name,
            thinking=self.ctx.model_config.thinking,
            output_mode=self.ctx.model_config.output_mode,
        )

        asyncio.run(self._analyze_scenes(args))

    async def _analyze_scenes(self, args: argparse.Namespace) -> None:
        """Analyze each scene group."""
        analysis_agent = SceneAnalysisAgent.create(args, self.ctx.model_config)

        scenes_dir = self.ctx.output_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)

        scenes_metadata: list[FilmSceneMetadata] = []

        cache_dir = self.ctx.output_dir / "cache"
        frame_base_path = cache_dir  # frame.frame_path is relative to cache_dir

        for scene_num, group in enumerate(self._shot_groups, 1):
            scene_id = f"fs{scene_num:03d}"

            print(f"  Analyzing scene {scene_id} ({len(group.shots)} shots) ...")

            first_shot_id = group.shots[0]
            last_shot_id = group.shots[-1]

            first_idx = int(first_shot_id.replace("sh", "")) - 1
            last_idx = int(last_shot_id.replace("sh", "")) - 1

            if self.ctx.shot_list is None or self.ctx.keyframes is None:
                continue

            first_shot = self.ctx.shot_list.get_shot(first_shot_id)
            last_shot = self.ctx.shot_list.get_shot(last_shot_id)

            if first_shot is None or last_shot is None:
                continue

            keyframes_for_scene = self.ctx.keyframes[first_idx:last_idx + 1]

            all_frames: list[Frame] = []
            for kf in keyframes_for_scene:
                all_frames.extend(kf.extracted_frames)

            max_frames = self.ctx.keyframe_extractor.config.max_frames_per_scene
            selected_frames = self.ctx.keyframe_extractor.select_keyframes_for_scene(
                all_frames, max_frames=max_frames, base_path=cache_dir
            )

            scene_start = first_shot.start_time
            scene_end = last_shot.end_time

            prompt = analysis_agent.build_analysis_prompt(
                scene_group=group,
                shot_list=self.ctx.shot_list,
                selected_frames=selected_frames,
                transcript=self.ctx.transcript or Transcript(language="unknown", entries=[]),
                sound_events=self.ctx.sound_events or [],
                music=self.ctx.music_segments or [],
                speaker_bank_manager=self.ctx.speaker_bank,
                project_name=self.ctx.project.name,
                scene_id=scene_id,
                frame_base_path=frame_base_path,
            )

            try:
                content = await analysis_agent.run(prompt)

                for spk_id, name in content.speaker_map_update.items():
                    if self.ctx.speaker_bank.speaker_bank.get_speaker(spk_id) is not None:
                        self.ctx.speaker_bank.confirm_speaker(spk_id, name, scene_id)
                    else:
                        print(f"  [Warning] VLM confirmed unknown speaker {spk_id} → {name}, skipping")

                scene_file_name = f"scene_{scene_id}.txt"
                scene_file_path = scenes_dir / scene_file_name

                scene_text = analysis_agent.format_scene_file(
                    scene_id=scene_id,
                    source_file=self.ctx.shot_list.video_file,
                    scene_start=scene_start,
                    scene_end=scene_end,
                    content=content,
                )
                scene_file_path.write_text(scene_text, encoding="utf-8")

                scene_metadata = FilmSceneMetadata(
                    scene_id=scene_id,
                    source={
                        "type": "film",
                        "file": self.ctx.shot_list.video_file,
                        "shots": group.shots,
                        "start_time": scene_start,
                        "end_time": scene_end,
                    },
                    content_file=str(scene_file_path.relative_to(self.ctx.output_dir)),
                    token_estimate=estimate_tokens(scene_text),
                    location=content.location,
                    time_of_day=content.time_of_day,
                    characters={
                        "present": content.characters_present,
                        "mentioned": content.characters_mentioned,
                    },
                    audio={
                        "has_music": True if self.ctx.music_segments else False,
                        "music_type": self.ctx.music_segments[0].music_type if self.ctx.music_segments else None,
                        "notable_sounds": [e.description for e in (self.ctx.sound_events or [])[:5]],
                    },
                    speaker_map_snapshot=self.ctx.speaker_bank.speaker_bank.speaker_map.copy(),
                )
                scenes_metadata.append(scene_metadata)

            except Exception as e:
                print(f"  Error analyzing scene {scene_id}: {e}", file=sys.stderr)
                continue

        import datetime

        self.ctx.scene_index = FilmSceneIndex(
            metadata={
                "version": "1.0",
                "generated_at": datetime.datetime.now().isoformat(),
                "project": self.ctx.project.name,
                "source_file": self.ctx.shot_list.video_file if self.ctx.shot_list else "",
                "source_type": "film",
                "total_scenes": len(scenes_metadata),
            },
            scenes=scenes_metadata,
        )

        self.ctx.speaker_bank.save(cache_dir)

    def _write_index(self) -> None:
        """Write the scene index to disk."""
        if self.ctx.scene_index is None:
            return

        index_path = self.ctx.output_dir / "scene_index.json"
        self.ctx.scene_index.save(index_path)
        print(f"\nWritten scene index: {index_path}")
        print(f"Total scenes: {len(self.ctx.scene_index.scenes)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured scenes from film/video sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.pipelines.film_scene_extraction --project projects/CPK --source cpk-film\n"
            "  uv run python -m yorishiro.pipelines.film_scene_extraction --source cpk-film --force\n"
        ),
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source ID within project",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing, ignore cache",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Shots per grouping batch",
    )

    from yorishiro.agent_utils import add_model_args
    add_model_args(parser)

    args = parser.parse_args()

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    if source_config.type != "film":
        print(f"Error: Source '{args.source}' is not a film source (type: {source_config.type})", file=sys.stderr)
        sys.exit(1)

    config = FilmExtractionConfig(batch_size=args.batch_size)
    model_config = project.resolved_model_config("scene")

    pipeline = FilmSceneExtractionPipeline.create(
        project=project,
        source_id=args.source,
        config=config,
        model_config=model_config,
    )

    try:
        pipeline.run(force=args.force)
        print("\nFilm scene extraction complete.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()