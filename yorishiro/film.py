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
import sys
from pathlib import Path

from yorishiro.project import Project
from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig
from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig
from yorishiro.audio.speech_pipeline import SpeechPipeline, SpeechPipelineConfig
from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig
from yorishiro.audio.sound_event_detector import SoundEventDetector, SoundEventDetectorConfig
from yorishiro.audio.music_analyzer import MusicAnalyzer, MusicAnalyzerConfig
from yorishiro.models.film_models import ShotList


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
        threshold=args.threshold,
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

    print(f"Extracting keyframes from {len(shot_list.shots)} shots...")
    print(f"Output format: {args.format}")

    config = KeyFrameExtractorConfig(
        frames_per_shot=args.frames_per_shot,
        max_frames_per_scene=args.max_frames_scene,
        output_format=args.format,
        output_quality=args.quality,
    )
    extractor = KeyFrameExtractor(config)

    keyframes = extractor.extract(video_path, shot_list, output_dir, force=args.force)

    print(f"\nExtracted frames from {len(keyframes)} shots")
    total_frames = sum(len(kf.extracted_frames) for kf in keyframes)
    print(f"Total frames: {total_frames}")


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

    speaker_bank = SpeakerBankManager(SpeakerBankManagerConfig())
    speaker_bank.load(output_dir)
    speaker_bank.save(output_dir)
    print(f"Speaker bank: {len(speaker_bank.speaker_bank.speakers)} speakers")

    audio_path = output_dir / "audio.wav"
    if not audio_path.exists():
        print("Warning: audio.wav not found, skipping sound/music analysis")
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
    print("Shot grouping requires LLM. Use 'extract' command for full pipeline.")
    print("For individual step control, use the Python API directly.")


def cmd_analyze_scenes(args: argparse.Namespace) -> None:
    """Step 2: Analyze scenes using VLM."""
    print("Scene analysis requires VLM. Use 'extract' command for full pipeline.")
    print("For individual step control, use the Python API directly.")


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

    # detect-shots
    p_detect = subparsers.add_parser("detect-shots", help="Layer 0A Step 1: Detect shots in video")
    p_detect.add_argument("--project", type=Path, default=None)
    p_detect.add_argument("--source", type=str, required=True)
    p_detect.add_argument("--force", action="store_true")
    p_detect.add_argument("--threshold", type=float, default=3.0)
    p_detect.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    # extract-frames
    p_frames = subparsers.add_parser("extract-frames", help="Layer 0A Step 2: Extract keyframes")
    p_frames.add_argument("--project", type=Path, default=None)
    p_frames.add_argument("--source", type=str, required=True)
    p_frames.add_argument("--force", action="store_true")
    p_frames.add_argument("--frames-per-shot", type=int, default=5)
    p_frames.add_argument("--max-frames-scene", type=int, default=8)
    p_frames.add_argument("--format", type=str, default="avif", choices=["avif", "jpg", "png"],
                         help="Output image format (default: avif)")
    p_frames.add_argument("--quality", type=int, default=85,
                         help="Output quality 1-100 (default: 85)")

    # process-audio
    p_audio = subparsers.add_parser("process-audio", help="Layer 0B: Process audio")
    p_audio.add_argument("--project", type=Path, default=None)
    p_audio.add_argument("--source", type=str, required=True)
    p_audio.add_argument("--force", action="store_true")
    p_audio.add_argument("--language", type=str, default=None)

    # group-shots
    p_group = subparsers.add_parser("group-shots", help="Step 1: Group shots into scenes (needs --model)")
    p_group.add_argument("--project", type=Path, default=None)
    p_group.add_argument("--source", type=str, required=True)
    p_group.add_argument("--force", action="store_true")
    p_group.add_argument("--batch-size", type=int, default=15)

    # analyze-scenes
    p_analyze = subparsers.add_parser("analyze-scenes", help="Step 2: Analyze scenes (needs --model)")
    p_analyze.add_argument("--project", type=Path, default=None)
    p_analyze.add_argument("--source", type=str, required=True)
    p_analyze.add_argument("--force", action="store_true")

    # extract (full pipeline)
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