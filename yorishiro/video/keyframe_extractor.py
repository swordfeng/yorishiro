"""Keyframe extraction using PyAV.

Extracts representative frames from video shots using 4-layer filtering:
1. Dense temporal sampling (1.6fps)
2. Pixel diversity filter (reduce candidates)
3. CLIP embedding + semantic selection
4. Save selected frames + embeddings

Used for:
1. Shot grouping (1 frame per shot - first frame)
2. Scene analysis (2-8 frames per shot, semantically diverse)
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
    model: str = Field(default="ViT-B/32", description="CLIP model variant (faster)")
    sampling_fps: float = Field(default=1.6, description="Frames per second for dense sampling")
    min_frames_per_shot: int = Field(default=2, description="Minimum frames to extract per shot")
    max_frames_per_shot: int = Field(default=8, description="Maximum frames to extract per shot")
    max_frames_per_scene: int = Field(default=8, description="Max frames per scene to give to VLM")
    output_format: str = Field(default="avif", description="Output image format: 'avif', 'jpg', or 'png'")
    output_quality: int = Field(default=85, description="Output quality (1-100)")
    workers: int = Field(default_factory=lambda: os.cpu_count() or 4, description="Worker threads for parallel frame processing")


class KeyFrameExtractor:
    """Extracts keyframes from video shots using 4-layer filtering + CLIP."""

    def __init__(self, config: KeyFrameExtractorConfig | None = None):
        self.config = config or KeyFrameExtractorConfig()
        self._clip_model = None
        self._processor = None
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
        CPU-bound work (thumbnail scoring, diversity selection, CLIP, image encoding) runs
        in parallel on a thread pool.
        """
        from tqdm import tqdm

        fps = float(self._video_stream.average_rate)
        futures: list[tuple[str, Future[tuple[list[Frame], np.ndarray]]]] = []

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
        results: dict[str, tuple[list[Frame], np.ndarray]] = {}

        for shot_id, future in futures:
            frames, embeddings = future.result()
            if frames:
                results[shot_id] = (frames, embeddings)

        for shot_id in sorted(results, key=lambda sid: shot_order[sid]):
            frames, embeddings = results[shot_id]
            keyframe_sets.append(KeyFrameSet(
                shot_id=shot_id,
                representative_frame=frames[0],
                extracted_frames=frames,
            ))
            # Save embeddings per shot
            shot_frames_dir = frames_dir / shot_id
            embeddings_path = shot_frames_dir / "embeddings.npz"
            np.savez_compressed(embeddings_path, 
                               **{f.frame_path.split("/")[-1]: emb for f, emb in zip(frames, embeddings)})

        return keyframe_sets

    def _decode_candidates(self, shot: Shot) -> list[tuple]:
        """Seek and decode candidate frames for one shot using 1.6fps dense sampling.

        Layer 1: Dense temporal sampling at 1.6fps. Fill to max_frames_per_shot if less.
        """
        duration = shot.end_time - shot.start_time
        num_candidates = int(duration * self.config.sampling_fps)
        num_candidates = max(self.config.max_frames_per_shot, num_candidates)

        timestamps = np.linspace(shot.start_time, shot.end_time, num_candidates + 2)[1:-1]
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
    ) -> tuple[list[Frame], np.ndarray]:
        """Process candidates through 5 layers.

        Layer 2: Pixel diversity filter (reduce candidates).
        Layer 3: CLIP embedding + semantic diversity → target.
        Layer 4: Semantic selection (CLIP diversity).
        Layer 5: Save selected frames + embeddings.

        Runs in thread pool.
        """
        min_frames = self.config.min_frames_per_shot
        max_frames = self.config.max_frames_per_shot

        # Layer 2: Filter 1 - Pixel diversity if we have more candidates than max_frames * 1.25
        max_survivors = int(max_frames * 1.25)
        if len(candidates) > max_survivors:
            survivors_count = min(max_survivors, len(candidates))
            filter1_indices = self._select_diverse_frames(
                [img for img, _ in candidates], survivors_count
            )
            candidates = [candidates[i] for i in filter1_indices]

        # Layer 3: Compute CLIP embeddings and determine semantic target
        embeddings = self._compute_embeddings_batch([img for img, _ in candidates])
        target = self._compute_semantic_target(embeddings, min_frames, max_frames)

        # Clamp target to available candidates
        target = min(target, len(candidates))

        # Layer 4: Semantic selection using precomputed embeddings
        selected_indices = self._semantic_select_from_embeddings(
            embeddings, target
        )

        # Layer 5: Save selected frames
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

        return frames, embeddings[selected_indices]

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
        """Select `count` indices via greedy max-min pixel diversity (Layer 2)."""
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

    def _compute_semantic_target(self, embeddings: np.ndarray, min_frames: int, max_frames: int) -> int:
        """Compute target frame count based on semantic diversity.

        Uses pairwise cosine similarity to determine how diverse the frames are.
        High diversity → more frames needed. Low diversity → fewer frames.

        Args:
            embeddings: CLIP embeddings array (N, 512)
            min_frames: Minimum frames to extract
            max_frames: Maximum frames to extract

        Returns:
            Target frame count in [min_frames, max_frames]
        """
        n = len(embeddings)
        if n < 2:
            return min_frames

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Compute pairwise cosine similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)
        np.fill_diagonal(sim_matrix, 0)  # exclude self-similarity

        # Average similarity (excluding diagonal)
        avg_sim = sim_matrix.sum() / (n * (n - 1))

        # Semantic diversity: 0.0 (identical) → 1.0 (completely different)
        semantic_diversity = 1.0 - avg_sim ** 2.2

        # Map to target range
        target = min_frames + round((max_frames - min_frames) * semantic_diversity)
        return max(min_frames, min(max_frames, target))

    def _semantic_select_from_embeddings(self, embeddings: np.ndarray, target: int) -> list[int]:
        """Select frames using precomputed CLIP embeddings.

        Always includes first and last frame as temporal anchors.
        Selects remaining frames by CLIP embedding diversity.

        Args:
            embeddings: CLIP embeddings array (N, 512)
            target: Number of frames to select

        Returns:
            List of selected indices
        """
        n = len(embeddings)
        if n <= 2:
            return list(range(n))

        if n <= target:
            return list(range(n))

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Select middle frames by CLIP diversity
        num_middle = target - 2  # exclude first and last
        middle_indices = list(range(1, n - 1))

        if num_middle <= 0:
            return [0, n - 1]

        # Greedy max-min diversity on middle indices
        selected_middle = [0]  # start with first middle frame index
        remaining_middle = set(range(1, len(middle_indices)))

        while len(selected_middle) < num_middle and remaining_middle:
            best_idx = None
            best_score = -1.0
            # Map selected_middle back to original indices
            selected_original = [middle_indices[i] for i in selected_middle]
            selected_original.extend([0, n - 1])  # include anchors
            sel_embeddings = normalized[selected_original]

            for idx in remaining_middle:
                original_idx = middle_indices[idx]
                sims = np.dot(sel_embeddings, normalized[original_idx])
                diversity = float(1.0 - np.max(sims))
                if diversity > best_score:
                    best_score = diversity
                    best_idx = idx

            if best_idx is not None:
                selected_middle.append(best_idx)
                remaining_middle.discard(best_idx)

        # Combine first, selected middle, and last
        result = [0]
        result.extend(sorted([middle_indices[i] for i in selected_middle]))
        result.append(n - 1)

        return result[:target]

    def _semantic_select(self, images: list, target: int) -> tuple[list[int], np.ndarray]:
        """Select frames using CLIP semantic diversity (Layer 3).

        Always includes first and last frame as temporal anchors.
        Selects remaining frames by CLIP embedding diversity.

        Returns:
            Tuple of (selected_indices, selected_embeddings)
        """
        if len(images) <= 2:
            # Return all frames with all embeddings
            embeddings = self._compute_embeddings_batch(images) if images else np.array([])
            return list(range(len(images))), embeddings

        # Always include first and last
        if len(images) <= target:
            embeddings = self._compute_embeddings_batch(images)
            return list(range(len(images))), embeddings

        # Compute embeddings for all images
        embeddings = self._compute_embeddings_batch(images)

        # Use precomputed embeddings for selection
        selected_indices = self._semantic_select_from_embeddings(embeddings, target)
        return selected_indices, embeddings[selected_indices]

    def _compute_embeddings_batch(self, images: list) -> np.ndarray:
        """Compute CLIP embeddings for a batch of images."""
        if self._clip_model is None:
            self._load_clip_model()

        import torch

        # Preprocess all images
        img_tensors = self._preprocess_images(images)

        if self._device is not None:
            img_tensors = img_tensors.to(self._device)

        with torch.no_grad():
            if self._clip_model is not None:
                output = self._clip_model.get_image_features(pixel_values=img_tensors)
                # get_image_features returns BaseModelOutputWithPooling
                # pooler_output contains the 512-dim CLIP image embedding
                embeddings = output.pooler_output
            else:
                raise RuntimeError("CLIP model not initialized")

        return embeddings.cpu().numpy()

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
        """Compute CLIP embedding for a single frame (for external use)."""
        if self._clip_model is None:
            self._load_clip_model()

        from PIL import Image
        import torch

        img = Image.open(frame_path).convert("RGB")
        img_tensor = self._preprocess_images([img])

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
        self._processor = processor

    def _preprocess_images(self, images: list) -> object:
        """Preprocess images for CLIP model."""

        if self._processor is None:
            raise RuntimeError("CLIP processor not initialized")

        inputs = self._processor(images=images, return_tensors="pt")
        if self._device is not None:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        return inputs["pixel_values"]

    def load_embeddings(self, shot_id: str, cache_dir: Path) -> np.ndarray | None:
        """Load cached CLIP embeddings for a shot.

        Args:
            shot_id: Shot identifier (e.g., "sh001")
            cache_dir: Cache directory containing frames/

        Returns:
            Embeddings array or None if not found
        """
        embeddings_path = cache_dir / "frames" / shot_id / "embeddings.npz"
        if not embeddings_path.exists():
            return None

        data = np.load(embeddings_path, allow_pickle=True)
        # Return stacked embeddings in order
        keys = sorted(data.files)
        return np.stack([data[k] for k in keys])

    def select_keyframes_for_scene(
        self,
        shot_frames: list[Frame],
        max_frames: int = 8,
        base_path: Path | None = None,
    ) -> list[Frame]:
        """Select keyframes from a list of shot frames using cached CLIP embeddings.

        This method is used for scene-level selection (Step 2).
        Loads cached embeddings instead of recomputing.
        """
        if len(shot_frames) <= max_frames:
            return shot_frames

        if base_path is None:
            # Can't load embeddings without base path
            return shot_frames[:max_frames]

        # Load cached embeddings
        embeddings = []
        valid_frames = []

        for frame in shot_frames:
            frame_path = base_path / frame.frame_path
            # Embeddings are cached in embeddings.npz next to frame files
            shot_dir = frame_path.parent
            embeddings_file = shot_dir / "embeddings.npz"

            if not embeddings_file.exists():
                # Fall back to computing
                if frame_path.exists():
                    try:
                        emb = self.compute_clip_embedding(frame_path)
                        embeddings.append(emb)
                        valid_frames.append(frame)
                    except Exception:
                        continue
                continue

            # Load from cache
            try:
                data = np.load(embeddings_file, allow_pickle=True)
                frame_key = frame_path.name
                if frame_key in data.files:
                    embeddings.append(data[frame_key])
                    valid_frames.append(frame)
            except Exception:
                continue

        if len(valid_frames) <= max_frames:
            return valid_frames

        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings_array / norms

        # Scene-level selection: max-min diversity across all frames from all shots
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