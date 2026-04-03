"""Shot detection using PySceneDetect.

Detects shot boundaries in video files and produces a ShotList.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataclasses import dataclass

from yorishiro.models.film_models import Shot, ShotList


@dataclass
class ShotDetectorConfig:
    backend: str = "pyscenedetect"
    detector: str = "adaptive"
    threshold: float = 4.0
    min_content_val: float = 15.0
    min_scene_len: int = 15
    show_progress: bool = True


class ShotDetector:
    """Detects shot boundaries in video files."""

    def __init__(self, config: ShotDetectorConfig | None = None):
        self.config = config or ShotDetectorConfig()

    @staticmethod
    def compute_video_hash(video_path: Path) -> str:
        """Compute a hash based on video file mtime + size."""
        stat = video_path.stat()
        data = f"{video_path.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def detect(self, video_path: Path, output_dir: Path, force: bool = False) -> ShotList:
        """Detect shots in video and return ShotList.

        Caches result to output_dir / shots.json. Use force=True to recompute.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_file = output_dir / "shots.json"

        video_hash = self.compute_video_hash(video_path)

        if not force and cache_file.exists():
            try:
                cached = ShotList(**json.loads(cache_file.read_text(encoding="utf-8")))
                if cached.video_hash == video_hash:
                    print(f"  [ShotDetector] Using cached shots: {len(cached.shots)} shots")
                    return cached
            except Exception:
                pass

        print(f"  [ShotDetector] Processing {video_path.name} ...")
        shots = self._detect_shots(video_path)

        result = ShotList(
            video_file=video_path.name,
            video_hash=video_hash,
            fps=shots.get("fps", 24.0),
            duration=shots.get("duration", 0.0),
            shots=[Shot(**s) for s in shots.get("shots", [])],
        )

        cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [ShotDetector] Detected {len(result.shots)} shots")
        return result

    def _detect_shots(self, video_path: Path) -> dict:
        """Run PySceneDetect and return shot data."""
        try:
            from scenedetect import detect, ContentDetector, AdaptiveDetector
        except ImportError:
            raise ImportError(
                "PySceneDetect not installed. Install with: pip install scenedetect[opencv]"
            )

        # Get video info first using PyAV
        fps = 24.0
        duration = 0.0
        total_frames = None

        try:
            import av
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]

            if video_stream.average_rate is not None:
                fps = float(video_stream.average_rate)

            # Try multiple ways to get duration
            if video_stream.duration and video_stream.time_base is not None:
                duration = float(video_stream.duration * video_stream.time_base)
            elif container.duration:
                duration = float(container.duration) / 1000000.0  # AV_TIME_BASE

            # Get frame count
            if video_stream.frames:
                total_frames = video_stream.frames

            container.close()
        except Exception as e:
            print(f"  [ShotDetector] Warning: Could not read video info with PyAV: {e}")

        print(f"  [ShotDetector] Video: {fps:.2f} fps, {duration:.1f}s" +
              (f", ~{total_frames} frames" if total_frames else ""))

        # Run shot detection with progress
        print("  [ShotDetector] Detecting shots...")

        if self.config.detector == "adaptive":
            detector = AdaptiveDetector(
                adaptive_threshold=self.config.threshold,
                min_content_val=self.config.min_content_val,
            )
        else:
            detector = ContentDetector(threshold=self.config.threshold)

        try:
            scene_list = detect(
                str(video_path),
                detector,
                show_progress=self.config.show_progress,
            )
        except TypeError:
            # Older version without show_progress
            scene_list = detect(str(video_path), detector)

        print(f"  [ShotDetector] Found {len(scene_list)} scene boundaries")

        valid_scenes = [
            (scene[0], scene[1])
            for scene in scene_list
            if self.config.min_scene_len == 0
            or scene[1].get_frames() - scene[0].get_frames() >= self.config.min_scene_len
        ]

        shots = []
        for i, (start, end) in enumerate(valid_scenes):
            shots.append({
                "shot_id": f"sh{i + 1:03d}",
                "start_time": start.get_seconds(),
                "end_time": end.get_seconds(),
                "start_frame": start.get_frames(),
                "end_frame": end.get_frames(),
            })

        return {
            "fps": fps,
            "duration": duration,
            "shots": shots,
        }