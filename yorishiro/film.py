"""CLI for film scene extraction with individual step commands.

Usage:
    uv run python -m yorishiro.film detect-shots --project <dir> --source <id>
    uv run python -m yorishiro.film extract-frames --project <dir> --source <id>
    uv run python -m yorishiro.film process-audio --project <dir> --source <id>
    uv run python -m yorishiro.film group-shots --project <dir> --source <id>
    uv run python -m yorishiro.film analyze-scenes --project <dir> --source <id>
    uv run python -m yorishiro.film extract --project <dir> --source <id>  # full pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from yorishiro.project import Project
from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig
from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig
from yorishiro.audio.speech_pipeline import SpeechPipeline, SpeechPipelineConfig
from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
from yorishiro.audio.sound_event_detector import SoundEventDetector, SoundEventDetectorConfig
from yorishiro.audio.music_analyzer import MusicAnalyzer, MusicAnalyzerConfig
from yorishiro.models.film_models import (
    ShotList,
    Transcript,
    SoundEvent,
    MusicSegment,
    ShotGroup,
    FilmSceneIndex,
)


def cmd_detect_shots(args: argparse.Namespace) -> None:
    """Layer 0A Step 1: Detect shots in video."""
    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    if source_config.type != "film":
        print(f"Error: Source '{args.source}' is not a film source", file=sys.stderr)
        sys.exit(1)

    video_path = project.get_source_path(args.source)
    output_dir = project.source_dir(args.source) / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ShotDetectorConfig(
        detector=args.detector,
        threshold=args.threshold,
        min_content_val=args.min_content_val,
        show_progress=not args.no_progress,
    )
    detector = ShotDetector(config)

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")

    shot_list = detector.detect(video_path, output_dir, force=args.force)

    print(f"\nDetected {len(shot_list.shots)} shots")
    print(f"Duration: {shot_list.duration:.1f}s")
    print(f"FPS: {shot_list.fps}")


def cmd_extract_frames(args: argparse.Namespace) -> None:
    """Layer 0A Step 2: Extract keyframes from shots."""
    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    video_path = project.get_source_path(args.source)
    output_dir = project.source_dir(args.source) / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    shots_file = output_dir / "shots.json"
    if not shots_file.exists():
        print("Error: shots.json not found. Run 'detect-shots' first.", file=sys.stderr)
        sys.exit(1)

    shot_list = ShotList.model_validate_json(shots_file.read_text(encoding="utf-8"))

    if args.shot:
        matching_shots = [s for s in shot_list.shots if s.shot_id == args.shot]
        if not matching_shots:
            print(f"Error: Shot '{args.shot}' not found. Available shots: {[s.shot_id for s in shot_list.shots[:5]]}...", file=sys.stderr)
            sys.exit(1)
        shot_list.shots = matching_shots
        print(f"Extracting keyframes from single shot: {args.shot}")
    else:
        print(f"Extracting keyframes from {len(shot_list.shots)} shots...")

    print(f"Output format: {args.format}")

    config = KeyFrameExtractorConfig(
        min_frames_per_shot=args.min_frames,
        max_frames_per_shot=args.max_frames,
        max_frames_per_scene=args.max_frames_scene,
        output_format=args.format,
        output_quality=args.quality,
        workers=args.workers,
    )
    extractor = KeyFrameExtractor(config)

    keyframes = extractor.extract(video_path, shot_list, output_dir, force=args.force)

    print(f"\nExtracted frames from {len(keyframes)} shots")
    total_frames = sum(len(kf.extracted_frames) for kf in keyframes)
    print(f"Total frames: {total_frames}")


def _resolve_speaker_ids(
    transcript: Transcript,
    speaker_bank: SpeakerBankManager,
    audio_path: Path,
    cache_dir: Path,
) -> None:
    """Resolve diarization local speaker IDs (SPEAKER_XX) to global SPKR_XXX IDs."""
    if not transcript.entries:
        return

    if all(e.speaker_global.startswith("SPKR_") for e in transcript.entries):
        return

    local_speakers: dict[str, Any] = {}
    for entry in transcript.entries:
        if entry.speaker_global not in local_speakers:
            local_speakers[entry.speaker_global] = entry

    local_to_global: dict[str, str] = {}
    for local_id, rep_entry in local_speakers.items():
        if audio_path.exists():
            embedding = speaker_bank.extract_speaker_embedding(
                audio_path, rep_entry.start, rep_entry.end
            )
        else:
            embedding = None
        global_id = speaker_bank.assign_global_speaker_id(
            local_id, embedding, rep_entry.start
        )
        local_to_global[local_id] = global_id
        print(f"    {local_id} → {global_id}")

    for entry in transcript.entries:
        entry.speaker_global = local_to_global.get(entry.speaker_global, entry.speaker_global)

    cache_file = cache_dir / "transcript.json"
    cache_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def cmd_process_audio(args: argparse.Namespace) -> None:
    """Layer 0B: Process audio (speech, sounds, music)."""
    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    video_path = project.get_source_path(args.source)
    output_dir = project.source_dir(args.source) / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    language = source_config.config.get("language") or args.language

    print("=== Speech Pipeline ===")
    speech_config = SpeechPipelineConfig(language=language if language != "auto" else None)
    speech_pipeline = SpeechPipeline(speech_config)
    transcript = speech_pipeline.process(video_path, output_dir, language=language, force=args.force)
    print(f"Transcribed {len(transcript.entries)} segments, language: {transcript.language}")

    print("\n=== Speaker Bank Management ===")
    speaker_bank = SpeakerBankManager(SpeakerBankManagerConfig())
    speaker_bank.load(output_dir)
    _resolve_speaker_ids(transcript, speaker_bank, output_dir / "audio.flac", output_dir)
    speaker_bank.save(output_dir)
    print(f"Speaker bank: {len(speaker_bank.speaker_bank.speakers)} speakers")

    audio_path = output_dir / "audio.flac"
    if not audio_path.exists():
        print("Warning: audio.flac not found, skipping sound/music analysis")
        return

    print("\n=== Sound Event Detection ===")
    sound_config = SoundEventDetectorConfig()
    sound_detector = SoundEventDetector(sound_config)
    transcript_end = transcript.entries[-1].end if transcript.entries else 0.0
    sound_events = sound_detector.detect(audio_path, output_dir, transcript_end=transcript_end, force=args.force)
    print(f"Detected {len(sound_events)} sound events")

    print("\n=== Music Analysis ===")
    music_config = MusicAnalyzerConfig()
    music_analyzer = MusicAnalyzer(music_config)
    music_segments = music_analyzer.analyze(video_path, audio_path, output_dir, force=args.force)
    print(f"Found {len(music_segments)} music segments")


def cmd_group_shots(args: argparse.Namespace) -> None:
    """Step 1: Group shots into scenes using LLM."""
    from yorishiro.agents.film.shot_grouping import ShotGroupingAgent, build_shot_audio_summary

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    cache_dir = project.source_dir(args.source) / "cache"
    if not cache_dir.exists():
        print("Error: cache directory not found. Run Layer 0A and 0B first.", file=sys.stderr)
        sys.exit(1)

    shots_file = cache_dir / "shots.json"
    if not shots_file.exists():
        print("Error: shots.json not found. Run 'detect-shots' first.", file=sys.stderr)
        sys.exit(1)

    transcript_file = cache_dir / "transcript.json"
    if not transcript_file.exists():
        print("Error: transcript.json not found. Run 'process-audio' first.", file=sys.stderr)
        sys.exit(1)

    sound_file = cache_dir / "sound_events.json"
    music_file = cache_dir / "music_analysis.json"

    shot_list = ShotList.model_validate_json(shots_file.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate_json(transcript_file.read_text(encoding="utf-8"))

    sound_events: list[SoundEvent] = []
    if sound_file.exists():
        try:
            data = json.loads(sound_file.read_text(encoding="utf-8"))
            sound_events = [SoundEvent(**e) for e in data.get("events", [])]
        except Exception:
            pass

    music_segments: list[MusicSegment] = []
    if music_file.exists():
        try:
            data = json.loads(music_file.read_text(encoding="utf-8"))
            music_segments = [MusicSegment(**m) for m in data.get("segments", [])]
        except Exception:
            pass

    frame_index_file = cache_dir / "frame_index.json"
    keyframes = []
    if frame_index_file.exists():
        try:
            from yorishiro.models.film_models import KeyFrameSet
            data = json.loads(frame_index_file.read_text(encoding="utf-8"))
            keyframes = [KeyFrameSet(**kf) for kf in data.get("keyframes", [])]
        except Exception:
            pass

    model_config = project.resolved_model_config("scene")
    model_args = argparse.Namespace(
        provider=model_config.provider,
        model=model_config.name,
        thinking=model_config.thinking,
        output_mode=model_config.output_mode,
    )

    grouping_agent = ShotGroupingAgent.create(model_args, model_config)

    shots = shot_list.shots
    batch_size = args.batch_size
    all_groups: list[ShotGroup] = []
    previous_summary = ""
    partial_group: ShotGroup | None = None
    idx = 0

    frame_base_path = cache_dir

    print(f"Grouping {len(shots)} shots into scenes (batch size: {batch_size})")

    while idx < len(shots):
        batch_end = min(idx + batch_size, len(shots))
        batch_shots = shots[idx:batch_end]
        batch_keyframes = keyframes[idx:batch_end]

        print(f"  Processing batch {idx // batch_size + 1}: shots {idx + 1}-{batch_end}")

        audio_summaries = []
        for shot in batch_shots:
            summary = build_shot_audio_summary(
                shot,
                transcript.entries,
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
            result = asyncio.run(grouping_agent.run(prompt))

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
            idx = max(result.next_shot_index, batch_end)

        except Exception as e:
            print(f"  Error in grouping: {e}", file=sys.stderr)
            idx = batch_end

    if partial_group:
        all_groups.append(partial_group)

    output_file = cache_dir / "shot_groups.json"
    output_data = {
        "total_shots": len(shots),
        "total_groups": len(all_groups),
        "groups": [g.model_dump() for g in all_groups],
    }
    output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGrouped {len(shots)} shots into {len(all_groups)} scenes")
    print(f"Output: {output_file}")


def cmd_analyze_scenes(args: argparse.Namespace) -> None:
    """Step 2: Analyze scenes using VLM."""
    from yorishiro.agent_utils import estimate_tokens
    from yorishiro.agents.film.scene_analysis import SceneAnalysisAgent
    from yorishiro.models.film_models import (
        Frame,
        KeyFrameSet,
        FilmSceneMetadata,
        Transcript,
        SoundEvent,
        MusicSegment,
    )

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
        sys.exit(1)

    cache_dir = project.source_dir(args.source) / "cache"
    if not cache_dir.exists():
        print("Error: cache directory not found. Run Layer 0A and 0B first.", file=sys.stderr)
        sys.exit(1)

    shot_groups_file = cache_dir / "shot_groups.json"
    if not shot_groups_file.exists():
        print("Error: shot_groups.json not found. Run 'group-shots' first.", file=sys.stderr)
        sys.exit(1)

    shots_file = cache_dir / "shots.json"
    if not shots_file.exists():
        print("Error: shots.json not found. Run 'detect-shots' first.", file=sys.stderr)
        sys.exit(1)

    frame_index_file = cache_dir / "frame_index.json"
    if not frame_index_file.exists():
        print("Error: frame_index.json not found. Run 'extract-frames' first.", file=sys.stderr)
        sys.exit(1)

    shot_list = ShotList.model_validate_json(shots_file.read_text(encoding="utf-8"))
    frame_data = json.loads(frame_index_file.read_text(encoding="utf-8"))
    keyframes = [KeyFrameSet(**kf) for kf in frame_data.get("keyframes", [])]

    shot_groups_data = json.loads(shot_groups_file.read_text(encoding="utf-8"))
    shot_groups = [ShotGroup(**g) for g in shot_groups_data.get("groups", [])]

    transcript = Transcript(language="unknown", entries=[])
    transcript_file = cache_dir / "transcript.json"
    if transcript_file.exists():
        try:
            transcript = Transcript.model_validate_json(transcript_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    sound_events: list[SoundEvent] = []
    sound_file = cache_dir / "sound_events.json"
    if sound_file.exists():
        try:
            data = json.loads(sound_file.read_text(encoding="utf-8"))
            sound_events = [SoundEvent(**e) for e in data.get("events", [])]
        except Exception:
            pass

    music_segments: list[MusicSegment] = []
    music_file = cache_dir / "music_analysis.json"
    if music_file.exists():
        try:
            data = json.loads(music_file.read_text(encoding="utf-8"))
            music_segments = [MusicSegment(**m) for m in data.get("segments", [])]
        except Exception:
            pass

    from yorishiro.audio.speaker_bank import SpeakerBankManager
    speaker_bank = SpeakerBankManager(SpeakerBankManagerConfig())
    speaker_bank.load(cache_dir)

    model_config = project.resolved_model_config("scene")
    model_args = argparse.Namespace(
        provider=model_config.provider,
        model=model_config.name,
        thinking=model_config.thinking,
        output_mode=model_config.output_mode,
    )

    analysis_agent = SceneAnalysisAgent.create(model_args, model_config)

    scenes_dir = project.source_dir(args.source) / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scenes_metadata: list[FilmSceneMetadata] = []
    frame_base_path = cache_dir
    extractor_config = KeyFrameExtractorConfig()

    for scene_num, group in enumerate(shot_groups, 1):
        scene_id = f"fs{scene_num:03d}"

        print(f"  Analyzing scene {scene_id} ({len(group.shots)} shots) ...")

        first_shot_id = group.shots[0]
        last_shot_id = group.shots[-1]

        first_idx = int(first_shot_id.replace("sh", "")) - 1
        last_idx = int(last_shot_id.replace("sh", "")) - 1

        first_shot = shot_list.get_shot(first_shot_id)
        last_shot = shot_list.get_shot(last_shot_id)

        if first_shot is None or last_shot is None:
            print(f"  [Warning] Could not find shots {first_shot_id} or {last_shot_id}, skipping")
            continue

        keyframes_for_scene = keyframes[first_idx:last_idx + 1]

        all_frames: list[Frame] = []
        for kf in keyframes_for_scene:
            all_frames.extend(kf.extracted_frames)

        max_frames = extractor_config.max_frames_per_scene
        selected_frames = KeyFrameExtractor(extractor_config).select_keyframes_for_scene(
            all_frames, max_frames=max_frames, base_path=cache_dir
        )

        scene_start = first_shot.start_time
        scene_end = last_shot.end_time

        prompt = analysis_agent.build_analysis_prompt(
            scene_group=group,
            shot_list=shot_list,
            selected_frames=selected_frames,
            transcript=transcript,
            sound_events=sound_events,
            music=music_segments,
            speaker_bank_manager=speaker_bank,
            project_name=project.name,
            scene_id=scene_id,
            frame_base_path=frame_base_path,
        )

        try:
            content = asyncio.run(analysis_agent.run(prompt))

            for spk_id, name in content.speaker_map_update.items():
                if speaker_bank.speaker_bank.get_speaker(spk_id) is not None:
                    speaker_bank.confirm_speaker(spk_id, name, scene_id)
                else:
                    print(f"  [Warning] VLM confirmed unknown speaker {spk_id} → {name}, skipping")

            scene_file_name = f"scene_{scene_id}.txt"
            scene_file_path = scenes_dir / scene_file_name

            scene_text = analysis_agent.format_scene_file(
                scene_id=scene_id,
                source_file=shot_list.video_file,
                scene_start=scene_start,
                scene_end=scene_end,
                content=content,
            )
            scene_file_path.write_text(scene_text, encoding="utf-8")

            scene_metadata = FilmSceneMetadata(
                scene_id=scene_id,
                source={
                    "type": "film",
                    "file": shot_list.video_file,
                    "shots": group.shots,
                    "start_time": scene_start,
                    "end_time": scene_end,
                },
                content_file=str(scene_file_path.relative_to(project.source_dir(args.source))),
                token_estimate=estimate_tokens(scene_text),
                location=content.location,
                time_of_day=content.time_of_day,
                characters={
                    "present": content.characters_present,
                    "mentioned": content.characters_mentioned,
                },
                audio={
                    "has_music": True if music_segments else False,
                    "music_type": music_segments[0].music_type if music_segments else None,
                    "notable_sounds": [e.description for e in sound_events[:5]],
                },
                speaker_map_snapshot=speaker_bank.speaker_bank.speaker_map.copy(),
            )
            scenes_metadata.append(scene_metadata)

        except Exception as e:
            print(f"  Error analyzing scene {scene_id}: {e}", file=sys.stderr)
            continue

    import datetime

    scene_index = FilmSceneIndex(
        metadata={
            "version": "1.0",
            "generated_at": datetime.datetime.now().isoformat(),
            "project": project.name,
            "source_file": shot_list.video_file,
            "source_type": "film",
            "total_scenes": len(scenes_metadata),
        },
        scenes=scenes_metadata,
    )

    index_path = project.source_dir(args.source) / "scene_index.json"
    scene_index.save(index_path)

    speaker_bank.save(cache_dir)

    print(f"\nAnalyzed {len(scenes_metadata)} scenes")
    print(f"Scene index: {index_path}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Run full extraction pipeline."""
    from yorishiro.pipelines.film_scene_extraction import FilmSceneExtractionPipeline, FilmExtractionConfig

    project_path = args.project if args.project else Path.cwd()
    project = Project.load(project_path)

    source_config = project.get_source(args.source)
    if not source_config:
        print(f"Error: Source '{args.source}' not found in project", file=sys.stderr)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Film scene extraction CLI with individual step commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_detect = subparsers.add_parser("detect-shots", help="Layer 0A Step 1: Detect shots in video")
    p_detect.add_argument("--project", type=Path, default=None)
    p_detect.add_argument("--source", type=str, required=True)
    p_detect.add_argument("--force", action="store_true")
    p_detect.add_argument("--detector", type=str, default="adaptive", choices=["adaptive", "content"],
                         help="Detector type (default: adaptive)")
    p_detect.add_argument("--threshold", type=float, default=4.0,
                         help="Threshold (adaptive: ratio vs local mean; content: absolute HSV delta)")
    p_detect.add_argument("--min-content-val", type=float, default=15.0,
                         help="AdaptiveDetector: minimum raw content score (default: 15.0)")
    p_detect.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    p_frames = subparsers.add_parser("extract-frames", help="Layer 0A Step 2: Extract keyframes")
    p_frames.add_argument("--project", type=Path, default=None)
    p_frames.add_argument("--source", type=str, required=True)
    p_frames.add_argument("--force", action="store_true")
    p_frames.add_argument("--shot", type=str, default=None,
                         help="Extract frames for a single shot (e.g., 'sh004')")
    p_frames.add_argument("--min-frames", type=int, default=2)
    p_frames.add_argument("--max-frames", type=int, default=8)
    p_frames.add_argument("--max-frames-scene", type=int, default=8)
    p_frames.add_argument("--format", type=str, default="avif", choices=["avif", "jpg", "png"],
                         help="Output image format (default: avif)")
    p_frames.add_argument("--quality", type=int, default=85,
                         help="Output quality 1-100 (default: 85)")
    p_frames.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                         help="Worker threads for parallel frame processing")

    p_audio = subparsers.add_parser("process-audio", help="Layer 0B: Process audio")
    p_audio.add_argument("--project", type=Path, default=None)
    p_audio.add_argument("--source", type=str, required=True)
    p_audio.add_argument("--force", action="store_true")
    p_audio.add_argument("--language", type=str, default=None)

    from yorishiro.agent_utils import add_model_args

    p_group = subparsers.add_parser("group-shots", help="Step 1: Group shots into scenes")
    p_group.add_argument("--project", type=Path, default=None)
    p_group.add_argument("--source", type=str, required=True)
    p_group.add_argument("--batch-size", type=int, default=15)
    add_model_args(p_group)

    p_analyze = subparsers.add_parser("analyze-scenes", help="Step 2: Analyze scenes with VLM")
    p_analyze.add_argument("--project", type=Path, default=None)
    p_analyze.add_argument("--source", type=str, required=True)
    add_model_args(p_analyze)

    p_extract = subparsers.add_parser("extract", help="Run full extraction pipeline")
    p_extract.add_argument("--project", type=Path, default=None)
    p_extract.add_argument("--source", type=str, required=True)
    p_extract.add_argument("--force", action="store_true")
    p_extract.add_argument("--batch-size", type=int, default=15)

    args = parser.parse_args()

    if args.command == "detect-shots":
        cmd_detect_shots(args)
    elif args.command == "extract-frames":
        cmd_extract_frames(args)
    elif args.command == "process-audio":
        cmd_process_audio(args)
    elif args.command == "group-shots":
        cmd_group_shots(args)
    elif args.command == "analyze-scenes":
        cmd_analyze_scenes(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
