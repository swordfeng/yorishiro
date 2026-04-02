"""CLI entry point for keyframe extraction.

Usage:
    uv run python -m yorishiro.video.keyframe_extractor --project <dir> --source <id> [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yorishiro.project import Project
from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig
from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig
from yorishiro.models.film_models import ShotList


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract keyframes from video shots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m yorishiro.video.keyframe_extractor --project projects/CPK --source cpk-film\n"
            "  uv run python -m yorishiro.video.keyframe_extractor --source cpk-film --min-frames 2 --max-frames 8\n"
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
        "--min-frames",
        type=int,
        default=2,
        help="Minimum frames to extract per shot",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Maximum frames to extract per shot",
    )
    parser.add_argument(
        "--skip-shot-detection",
        action="store_true",
        help="Skip shot detection, use existing shots.json",
    )

    args = parser.parse_args()

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

    shots_file = output_dir / "shots.json"

    if args.skip_shot_detection and shots_file.exists():
        print("Loading existing shots...")
        shot_list = ShotList(**json.loads(shots_file.read_text(encoding="utf-8")))
    else:
        print("Running shot detection...")
        shot_detector = ShotDetector(ShotDetectorConfig())
        shot_list = shot_detector.detect(video_path, output_dir, force=args.force)

    print(f"\nExtracting keyframes from {len(shot_list.shots)} shots...")

    config = KeyFrameExtractorConfig(min_frames_per_shot=args.min_frames, max_frames_per_shot=args.max_frames)
    extractor = KeyFrameExtractor(config)

    keyframes = extractor.extract(video_path, shot_list, output_dir, force=args.force)

    print(f"\nExtracted frames from {len(keyframes)} shots")
    total_frames = sum(len(kf.extracted_frames) for kf in keyframes)
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()