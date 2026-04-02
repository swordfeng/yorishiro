"""Keyframe extraction using PyAV.

Extracts representative frames from video shots for:
1. Shot grouping (1 frame per shot)
2. Scene analysis (3-5 frames per scene, CLIP-selected)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from yorishiro.models.film_models import Frame, KeyFrameSet, Shot, ShotList


class KeyFrameExtractorConfig(BaseModel):
    backend: str = Field(default="clip", description="Embedding backend")
    model: str = Field(default="ViT-L/14", description="CLIP model variant")
    min_frames_per_shot: int = Field(default=2, description="Minimum frames to extract per shot")
    max_frames_per_shot: int = Field(default=8, description="Maximum frames to extract per shot")
    max_frames_per_scene: int = Field(default=8, description="Max frames per scene to give to VLM")
    output_format: str = Field(default="avif", description="Output image format: 'avif', 'jpg', or 'png'")
    output_quality: int = Field(default=85, description="Output quality (1-100)")
    workers: int = Field(default_factory=lambda: os.cpu_count() or 4, description="Worker threads for parallel frame processing")


class KeyFrameExtractor:
    """Extracts keyframes from video shots using PyAV."""

    def __init__(self, config: KeyFrameExtractorConfig | None = None):
        self.config = config or KeyFrameExtractorConfig()
        self._clip_model = None
        self._preprocess = None
        self._device = None
        self._video_container = None
        self._video_stream = None

    def extract(
        self,
        video_path: Path,
        shot_list: ShotList,
        output_dir: Path,
        force: bool = False,
    ) -> list[KeyFrameSet]:
        """Extract keyframes for all shots."""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cache_file = output_dir / "frame_index.json"

        if not force and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("video_hash") == shot_list.video_hash:
                    print(f"  [KeyFrameExtractor] Using cached frames: {len(cached['keyframes'])} shots")
                    return [KeyFrameSet(**kf) for kf in cached["keyframes"]]
            except Exception:
                pass

        print(f"  [KeyFrameExtractor] Extracting frames from {video_path.name} ...")

        self._open_video(video_path)
        try:
            keyframe_sets = self._extract_all_frames(shot_list, frames_dir)
        finally:
            self._close_video()

        result_data = {
            "video_hash": shot_list.video_hash,
            "keyframes": [kf.model_dump() for kf in keyframe_sets],
        }
        cache_file.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  [KeyFrameExtractor] Extracted frames for {len(keyframe_sets)} shots")
        return keyframe_sets

    def _open_video(self, video_path: Path) -> None:
        """Open video file with PyAV."""
        try:
            import av
        except ImportError:
            raise ImportError("PyAV not installed. Install with: pip install av")

        self._video_container = av.open(str(video_path))
        self._video_stream = self._video_container.streams.video[0]
        self._video_stream.thread_type = "AUTO"

    def _close_video(self) -> None:
        """Close video file."""
        if self._video_container:
            self._video_container.close()
            self._video_container = None
            self._video_stream = None

    def _extract_all_frames(
        self, shot_list: ShotList, frames_dir: Path
    ) -> list[KeyFrameSet]:
        """Extract frames for all shots using a producer/consumer pipeline.

        Video seeks/decodes happen sequentially on the main thread (single container).
        CPU-bound work (thumbnail scoring, diversity selection, image encoding) runs
        in parallel on a thread pool.
        """
        from tqdm import tqdm

        fps = float(self._video_stream.average_rate)
        futures: list[tuple[str, Future[list[Frame]]]] = []

        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for shot in tqdm(shot_list.shots, desc="Decoding shots", unit="shot"):
                shot_frames_dir = frames_dir / shot.shot_id
                shot_frames_dir.mkdir(parents=True, exist_ok=True)

                candidates = self._decode_candidates(shot)
                if candidates:
                    future = pool.submit(
                        self._process_candidates,
                        candidates,
                        shot_frames_dir,
                        frames_dir.parent,
                        fps,
                    )
                    futures.append((shot.shot_id, future))

        keyframe_sets = []
        shot_order = {shot.shot_id: i for i, shot in enumerate(shot_list.shots)}
        results: dict[str, list[Frame]] = {}

        for shot_id, future in futures:
            frames = future.result()
            if frames:
                results[shot_id] = frames

        for shot_id in sorted(results, key=lambda sid: shot_order[sid]):
            frames = results[shot_id]
            keyframe_sets.append(KeyFrameSet(
                shot_id=shot_id,
                representative_frame=frames[0],
                extracted_frames=frames,
            ))

        return keyframe_sets

    def _decode_candidates(self, shot: Shot) -> list[tuple]:
        """Seek and decode candidate frames for one shot. Runs on main thread."""
        max_frames = self.config.max_frames_per_shot

        timestamps = np.linspace(shot.start_time, shot.end_time, max_frames + 2)[1:-1]
        if len(timestamps) < 1:
            timestamps = np.array([(shot.start_time + shot.end_time) / 2])

        candidates: list[tuple] = []
        for ts in timestamps:
            frame_pts = int(ts / self._video_stream.time_base)
            self._video_container.seek(frame_pts, stream=self._video_stream)
            for frame in self._video_container.decode(video=0):
                if frame.pts is not None and frame.pts >= frame_pts:
                    candidates.append((frame.to_image(), float(ts)))
                    break

        return candidates

    def _process_candidates(
        self,
        candidates: list[tuple],
        output_dir: Path,
        cache_dir: Path,
        fps: float,
    ) -> list[Frame]:
        """Score, select, and save frames for one shot. Runs in thread pool."""
        min_frames = self.config.min_frames_per_shot
        max_frames = self.config.max_frames_per_shot

        dynamics_score = self._compute_dynamics_score([img for img, _ in candidates])
        target = min_frames + round((max_frames - min_frames) * dynamics_score)
        target = max(min_frames, min(max_frames, min(target, len(candidates))))

        if target < len(candidates):
            selected_indices = self._select_diverse_frames(
                [img for img, _ in candidates], target
            )
        else:
            selected_indices = list(range(len(candidates)))

        ext = self.config.output_format
        frames = []
        for i, idx in enumerate(selected_indices):
            img, ts = candidates[idx]
            output_path = output_dir / f"frame_{i:03d}.{ext}"

            if ext == "avif":
                actual_path = self._save_avif(img, output_path)
            elif ext == "png":
                img.save(output_path, "PNG", compress_level=9)
                actual_path = output_path
            else:
                img.save(output_path, "JPEG", quality=self.config.output_quality)
                actual_path = output_path

            frames.append(Frame(
                frame_path=str(actual_path.relative_to(cache_dir)),
                timestamp=ts,
                frame_number=int(ts * fps),
            ))

        return frames

    def _compute_dynamics_score(self, images: list) -> float:
        """Compute visual dynamics score [0, 1] from a sequence of PIL images.

        Uses mean absolute difference between adjacent 64x64 grayscale thumbnails,
        saturating at 0.2 (20% pixel change per frame = fully dynamic).
        """
        if len(images) < 2:
            return 0.0

        thumbs = [
            np.array(img.resize((64, 64)).convert("L"), dtype=np.float32)
            for img in images
        ]

        diffs = [
            float(np.mean(np.abs(thumbs[i + 1] - thumbs[i])) / 255.0)
            for i in range(len(thumbs) - 1)
        ]

        raw = float(np.mean(diffs))
        # Static shots: ~0.01–0.05; dynamic shots: ~0.1–0.3+; saturate at 0.2
        return min(1.0, raw / 0.2)

    def _select_diverse_frames(self, images: list, count: int) -> list[int]:
        """Select `count` indices via greedy max-min pixel diversity."""
        if count >= len(images):
            return list(range(len(images)))

        vecs = []
        for img in images:
            flat = np.array(img.resize((64, 64)).convert("L"), dtype=np.float32).flatten()
            norm = np.linalg.norm(flat)
            vecs.append(flat / norm if norm > 0 else flat)
        vecs_array = np.array(vecs)

        selected = [0]
        remaining = set(range(1, len(images)))

        while len(selected) < count and remaining:
            best_idx = None
            best_score = -1.0
            sel_vecs = vecs_array[selected]
            for idx in remaining:
                sims = np.dot(sel_vecs, vecs_array[idx])
                diversity = float(1.0 - np.max(sims))
                if diversity > best_score:
                    best_score = diversity
                    best_idx = idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)

        selected.sort()  # preserve temporal order
        return selected

    def _save_avif(self, img, output_path: Path) -> Path:
        """Save image as AVIF. Falls back to JPEG if AVIF not supported."""
        saved_path = output_path

        try:
            import pillow_avif  # noqa: F401 - registers AVIF plugin
            img.save(output_path, "AVIF", quality=self.config.output_quality)
            return saved_path
        except ImportError:
            pass

        try:
            img.save(output_path, "AVIF", quality=self.config.output_quality)
            return saved_path
        except Exception:
            jpeg_path = output_path.with_suffix(".jpg")
            img.save(jpeg_path, "JPEG", quality=self.config.output_quality)
            return jpeg_path

    def compute_clip_embedding(self, frame_path: Path) -> np.ndarray:
        """Compute CLIP embedding for a single frame."""
        if self._clip_model is None:
            self._load_clip_model()

        from PIL import Image
        import torch

        img = Image.open(frame_path).convert("RGB")
        if self._preprocess is not None:
            img_tensor = self._preprocess(img).unsqueeze(0)
        else:
            raise RuntimeError("CLIP preprocess not initialized")

        if self._device is not None:
            img_tensor = img_tensor.to(self._device)

        with torch.no_grad():
            if self._clip_model is not None:
                embedding = self._clip_model.encode_image(img_tensor)
            else:
                raise RuntimeError("CLIP model not initialized")

        return embedding.cpu().numpy().flatten()

    def _load_clip_model(self) -> None:
        """Load CLIP model for embedding computation."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError(
                "transformers and torch required for CLIP. Install with: pip install transformers torch"
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = self.config.model.replace("ViT-L/14", "openai/clip-vit-large-patch14")
        model_name = model_name.replace("ViT-B/32", "openai/clip-vit-base-patch32")

        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self._clip_model = model.to(self._device)
        self._preprocess = processor.images

    def select_keyframes_for_scene(
        self,
        shot_frames: list[Frame],
        max_frames: int = 8,
        base_path: Path | None = None,
    ) -> list[Frame]:
        """Select keyframes from a list of shot frames using CLIP diversity."""
        if len(shot_frames) <= max_frames:
            return shot_frames

        embeddings = []
        valid_frames = []
        for frame in shot_frames:
            try:
                frame_path = base_path / frame.frame_path if base_path else Path(frame.frame_path)
                if frame_path.exists():
                    emb = self.compute_clip_embedding(frame_path)
                    embeddings.append(emb)
                    valid_frames.append(frame)
            except Exception:
                continue

        if len(valid_frames) <= max_frames:
            return valid_frames

        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings_array / norms

        selected_indices = [0]
        remaining = set(range(1, len(valid_frames)))

        while len(selected_indices) < max_frames and remaining:
            best_idx = None
            best_score = -1.0

            selected_embeddings = normalized[selected_indices]

            for idx in remaining:
                # Max-min diversity: min similarity to any already-selected frame
                sims = np.dot(selected_embeddings, normalized[idx])
                diversity = float(1.0 - np.max(sims))

                if diversity > best_score:
                    best_score = diversity
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.discard(best_idx)

        selected_indices.sort(key=lambda i: valid_frames[i].timestamp)
        return [valid_frames[i] for i in selected_indices[:max_frames]]