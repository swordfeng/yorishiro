"""Step-scoped model runtime registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

from pydantic_ai import Agent

from yorishiro.agent_utils import build_agent_from_config

if TYPE_CHECKING:
    from yorishiro.project import Project

_OutputT = TypeVar("_OutputT")


@dataclass(frozen=True)
class _RuntimeSpec:
    kind: Literal["agent", "instance"]
    instance_builder: Callable[[ModelRegistry], Any] | None = None
    cache_key_builder: Callable[[ModelRegistry], str] | None = None


class StepRuntime:
    """Runtime facade for a single step's model dependency."""

    def __init__(
        self, registry: ModelRegistry, step_id: str, spec: _RuntimeSpec
    ) -> None:
        self._registry = registry
        self._step_id = step_id
        self._spec = spec

    def agent(
        self,
        *,
        output_type: type[_OutputT],
        system_prompt: str,
        tools: list | None = None,
    ) -> Agent[None, _OutputT]:
        if self._spec.kind != "agent":
            raise TypeError(
                f"Step '{self._step_id}' does not support agent() runtime access; "
                f"use instance() instead."
            )
        return self._registry._build_agent_for_step(
            self._step_id,
            output_type=output_type,
            system_prompt=system_prompt,
            tools=tools,
        )

    def instance(self) -> Any:
        if self._spec.kind != "instance":
            raise TypeError(
                f"Step '{self._step_id}' does not support instance() runtime access; "
                f"use agent() instead."
            )
        return self._registry._build_instance_for_step(self._step_id, self._spec)


class ModelRegistry:
    """Resolves step-scoped model runtimes and caches local model instances."""

    _STEP_SPECS: dict[str, _RuntimeSpec] = {
        "novel.scenes": _RuntimeSpec(kind="agent"),
        "novel.aliases": _RuntimeSpec(kind="agent"),
        "novel.characters": _RuntimeSpec(kind="agent"),
        "film.shot_groups": _RuntimeSpec(kind="agent"),
        "film.scenes": _RuntimeSpec(kind="agent"),
        "cross.synthesize": _RuntimeSpec(kind="agent"),
        "film.shots": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_shot_detector(),
            cache_key_builder=lambda registry: registry._cache_key_for_step(
                "film.shots", "model", "detector", "backend", fallback="shot-detector"
            ),
        ),
        "film.frames": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_keyframe_extractor(),
            cache_key_builder=lambda registry: registry._cache_key_for_step(
                "film.frames",
                "model",
                "backend",
                fallback="keyframe-extractor",
            ),
        ),
        "film.audio.separate": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_audio_separator(),
            cache_key_builder=lambda registry: registry._audio_separator_cache_key(),
        ),
        "film.audio.vad": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_vad_runner(),
            cache_key_builder=lambda registry: registry._cache_key_for_step(
                "film.audio.vad",
                "backend",
                "profile",
                fallback="vad-runner",
            ),
        ),
        "film.audio.stt": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_transcriber(),
            cache_key_builder=lambda registry: registry._transcriber_cache_key(),
        ),
        "film.audio.speakers": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_speaker_attributor(),
            cache_key_builder=lambda registry: registry._speaker_attributor_cache_key(),
        ),
        "film.audio.emotion": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_emotion_analyzer(),
            cache_key_builder=lambda registry: registry._emotion_analyzer_cache_key(),
        ),
        "film.audio.sound_events": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_sound_event_detector(),
            cache_key_builder=lambda registry: registry._cache_key_for_step(
                "film.audio.sound_events", fallback="sound-events"
            ),
        ),
        "film.audio.music": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_music_analyzer(),
            cache_key_builder=lambda registry: registry._music_analyzer_cache_key(),
        ),
    }

    def __init__(self, project: Project) -> None:
        self._project = project
        self._cache: dict[str, Any] = {}

    def release_all(self) -> None:
        for instance in self._cache.values():
            release_fn = getattr(instance, "release_models", None)
            if callable(release_fn):
                release_fn()
        self._cache.clear()
        from yorishiro.audio._speech_support import release_model_caches

        release_model_caches()

    def for_step(self, step_id: str) -> StepRuntime:
        """Return a runtime facade for the given step."""
        spec = self._STEP_SPECS.get(step_id)
        if spec is None:
            raise ValueError(f"No model runtime registered for step '{step_id}'")
        return StepRuntime(self, step_id, spec)

    def _build_agent_for_step(
        self,
        step_id: str,
        *,
        output_type: type[_OutputT],
        system_prompt: str,
        tools: list | None = None,
    ) -> Agent[None, _OutputT]:
        config = self._project.resolved_model_config(step_id)
        return build_agent_from_config(
            config,
            output_type=output_type,
            system_prompt=system_prompt,
            tools=tools,
        )

    def _build_instance_for_step(self, step_id: str, spec: _RuntimeSpec) -> Any:
        if spec.instance_builder is None or spec.cache_key_builder is None:
            raise TypeError(
                f"Step '{step_id}' is missing instance runtime builder configuration"
            )
        cache_key = spec.cache_key_builder(self)
        if cache_key not in self._cache:
            self._cache[cache_key] = spec.instance_builder(self)
        return self._cache[cache_key]

    def _cache_key_for_step(self, step_id: str, *fields: str, fallback: str) -> str:
        cfg = self._project.step_config(step_id)
        values = [
            str(cfg[field])
            for field in fields
            if field in cfg and cfg[field] is not None
        ]
        identity = "|".join(values) if values else fallback
        return f"{step_id}::{identity}"

    def _audio_separator_cache_key(self) -> str:
        return self._cache_key_for_step(
            "film.audio.separate",
            "backend",
            "model",
            "device",
            "sample_rate",
            "processing_chunk_seconds",
            "processing_overlap_seconds",
            fallback="audio-separator",
        )

    def _transcriber_cache_key(self) -> str:
        cfg = self._project.step_config("film.audio.stt")
        parts = [
            "transcriber",
            str(cfg.get("backend", "")),
            str(cfg.get("model", "")),
            str(cfg.get("cpu_threads", "")),
            str(cfg.get("num_workers", "")),
            str(cfg.get("word_timestamps", "")),
            str(cfg.get("vad_filter", "")),
            str(cfg.get("vad_min_silence_duration_ms", "")),
            str(cfg.get("checkpoint_shard_size", "")),
            str(cfg.get("group_max_duration_seconds", "")),
            str(cfg.get("group_max_gap_seconds", "")),
            str(cfg.get("min_confidence", "")),
            str(cfg.get("max_chars_per_second", "")),
            str(cfg.get("min_segment_seconds", "")),
            str(cfg.get("language", "")),
            str(cfg.get("forced_aligner_enabled", "")),
            str(cfg.get("forced_aligner_backend", "")),
            str(cfg.get("forced_aligner_model", "")),
            str(cfg.get("forced_aligner_device", "")),
            str(cfg.get("forced_aligner_num_workers", "")),
            str(cfg.get("forced_aligner_min_confidence", "")),
            str(cfg.get("forced_aligner_merge_gap_seconds", "")),
        ]
        return "film.audio.stt::" + "|".join(parts)

    def _speaker_attributor_cache_key(self) -> str:
        cfg = self._project.step_config("film.audio.speakers")
        parts = [
            "speaker-attributor",
            str(cfg.get("backend", "")),
            str(cfg.get("speaker_similarity_threshold", "")),
            str(cfg.get("diagnostics_enabled", "")),
            str(cfg.get("clustering_method", "")),
            str(cfg.get("window_duration", "")),
            str(cfg.get("window_hop", "")),
            str(cfg.get("min_window_duration", "")),
            str(cfg.get("energy_threshold_db", "")),
            str(cfg.get("norm_filter_sigma", "")),
            str(cfg.get("coherence_threshold", "")),
            str(cfg.get("min_vote_similarity", "")),
            str(cfg.get("hdbscan_min_cluster_size", "")),
            str(cfg.get("umap_n_neighbors", "")),
            str(cfg.get("umap_min_dist", "")),
            str(cfg.get("umap_n_components", "")),
            str(cfg.get("utterance_aggregation", "")),
        ]
        return "film.audio.speakers::" + "|".join(parts)

    def _emotion_analyzer_cache_key(self) -> str:
        cfg = self._project.step_config("film.audio.emotion")
        return f"film.audio.emotion::{cfg.get('model', 'emotion2vec/emotion2vec_plus_base')}"

    def _music_analyzer_cache_key(self) -> str:
        cfg = self._project.step_config("film.audio.music")
        separation = cfg.get("separation", {})
        analysis = cfg.get("analysis", {})
        if not isinstance(separation, dict):
            separation = {}
        if not isinstance(analysis, dict):
            analysis = {}
        parts = [
            "music-analyzer",
            str(separation.get("backend", "")),
            str(separation.get("model", "")),
            str(analysis.get("backend", "")),
            str(analysis.get("model", "")),
            str(cfg.get("detect_lyrics", "")),
        ]
        return "film.audio.music::" + "|".join(parts)

    def _build_shot_detector(self) -> Any:
        from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig

        cfg = self._project.step_config("film.shots")
        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["detector"] = cfg["backend"]
        elif cfg.get("detector") is not None:
            kwargs["detector"] = cfg["detector"]
        if cfg.get("threshold") is not None:
            kwargs["threshold"] = float(cfg["threshold"])
        if cfg.get("min_content_val") is not None:
            kwargs["min_content_val"] = float(cfg["min_content_val"])
        return ShotDetector(ShotDetectorConfig(**kwargs))

    def _build_keyframe_extractor(self) -> Any:
        from yorishiro.video.keyframe_extractor import (
            KeyFrameExtractor,
            KeyFrameExtractorConfig,
        )

        cfg = self._project.step_config("film.frames")
        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["backend"] = cfg["backend"]
        if cfg.get("model") is not None:
            kwargs["model"] = cfg["model"]
        elif cfg.get("clip_model") is not None:
            kwargs["model"] = cfg["clip_model"]
        if cfg.get("min_frames_per_shot") is not None:
            kwargs["min_frames_per_shot"] = int(cfg["min_frames_per_shot"])
        if cfg.get("max_frames_per_shot") is not None:
            kwargs["max_frames_per_shot"] = int(cfg["max_frames_per_shot"])
        if cfg.get("max_frames_per_scene") is not None:
            kwargs["max_frames_per_scene"] = int(cfg["max_frames_per_scene"])
        if cfg.get("output_format") is not None:
            kwargs["output_format"] = cfg["output_format"]
        if cfg.get("output_quality") is not None:
            kwargs["output_quality"] = int(cfg["output_quality"])
        return KeyFrameExtractor(KeyFrameExtractorConfig(**kwargs))

    def _build_vad_runner(self) -> Any:
        from yorishiro.audio.vad import VadConfig, VadRunner

        cfg = self._project.step_config("film.audio.vad")
        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["vad_backend"] = cfg["backend"]
        elif cfg.get("vad_backend") is not None:
            kwargs["vad_backend"] = cfg["vad_backend"]
        if cfg.get("profile") is not None:
            kwargs["vad_profile"] = cfg["profile"]
        elif cfg.get("vad_profile") is not None:
            kwargs["vad_profile"] = cfg["vad_profile"]
        return VadRunner(VadConfig(**kwargs))

    def _build_transcriber(self) -> Any:
        from yorishiro.audio.transcription import Transcriber, TranscriberConfig

        cfg = self._project.step_config("film.audio.stt")
        kwargs: dict[str, Any] = {}

        def _parse_bool(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
            if v is None:
                return False
            return bool(v)

        if cfg.get("backend") is not None:
            kwargs["stt_backend"] = cfg["backend"]
        if cfg.get("model") is not None:
            kwargs["stt_model"] = cfg["model"]
        if cfg.get("cpu_threads") is not None:
            kwargs["stt_cpu_threads"] = int(cfg["cpu_threads"])
        if cfg.get("num_workers") is not None:
            kwargs["stt_num_workers"] = int(cfg["num_workers"])
        if cfg.get("word_timestamps") is not None:
            kwargs["stt_word_timestamps"] = _parse_bool(cfg["word_timestamps"])
        if cfg.get("vad_filter") is not None:
            kwargs["stt_vad_filter"] = _parse_bool(cfg["vad_filter"])
        if cfg.get("vad_min_silence_duration_ms") is not None:
            kwargs["stt_vad_min_silence_duration_ms"] = int(
                cfg["vad_min_silence_duration_ms"]
            )
        if cfg.get("checkpoint_shard_size") is not None:
            kwargs["stt_checkpoint_shard_size"] = int(cfg["checkpoint_shard_size"])
        if cfg.get("group_max_duration_seconds") is not None:
            kwargs["stt_group_max_duration_seconds"] = float(
                cfg["group_max_duration_seconds"]
            )
        if cfg.get("group_max_gap_seconds") is not None:
            kwargs["stt_group_max_gap_seconds"] = float(cfg["group_max_gap_seconds"])
        if cfg.get("min_confidence") is not None:
            kwargs["stt_min_confidence"] = float(cfg["min_confidence"])
        if cfg.get("max_chars_per_second") is not None:
            kwargs["stt_max_chars_per_second"] = float(cfg["max_chars_per_second"])
        if cfg.get("min_segment_seconds") is not None:
            kwargs["stt_min_segment_seconds"] = float(cfg["min_segment_seconds"])
        if cfg.get("language") is not None:
            kwargs["language"] = cfg["language"]
        if cfg.get("extra_args") is not None:
            kwargs["stt_extra_args"] = dict(cfg["extra_args"])
        if cfg.get("forced_aligner_enabled") is not None:
            kwargs["forced_aligner_enabled"] = _parse_bool(
                cfg["forced_aligner_enabled"]
            )
        if cfg.get("forced_aligner_backend") is not None:
            kwargs["forced_aligner_backend"] = cfg["forced_aligner_backend"]
        if cfg.get("forced_aligner_model") is not None:
            kwargs["forced_aligner_model"] = cfg["forced_aligner_model"]
        if cfg.get("forced_aligner_device") is not None:
            kwargs["forced_aligner_device"] = cfg["forced_aligner_device"]
        if cfg.get("forced_aligner_num_workers") is not None:
            kwargs["forced_aligner_num_workers"] = int(
                cfg["forced_aligner_num_workers"]
            )
        if cfg.get("forced_aligner_min_confidence") is not None:
            kwargs["forced_aligner_min_confidence"] = float(
                cfg["forced_aligner_min_confidence"]
            )
        if cfg.get("forced_aligner_merge_gap_seconds") is not None:
            kwargs["forced_aligner_merge_gap_seconds"] = float(
                cfg["forced_aligner_merge_gap_seconds"]
            )
        if cfg.get("debug_dump_stt_diag") is not None:
            kwargs["debug_dump_stt_diag"] = _parse_bool(cfg["debug_dump_stt_diag"])

        return Transcriber(TranscriberConfig(**kwargs))

    def _build_speaker_attributor(self) -> Any:
        from yorishiro.audio.speaker_attribution import (
            SpeakerAttributor,
            SpeakerAttributorConfig,
        )

        cfg = self._project.step_config("film.audio.speakers")

        def _parse_bool(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
            if v is None:
                return False
            return bool(v)

        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["embedding_backend"] = cfg["backend"]
        if cfg.get("speaker_similarity_threshold") is not None:
            kwargs["similarity_threshold"] = float(cfg["speaker_similarity_threshold"])
        if cfg.get("diagnostics_enabled") is not None:
            kwargs["diagnostics_enabled"] = _parse_bool(cfg["diagnostics_enabled"])
        if cfg.get("clustering_method") is not None:
            kwargs["clustering_method"] = cfg["clustering_method"]
        if cfg.get("window_duration") is not None:
            kwargs["window_duration"] = float(cfg["window_duration"])
        if cfg.get("window_hop") is not None:
            kwargs["window_hop"] = float(cfg["window_hop"])
        if cfg.get("min_window_duration") is not None:
            kwargs["min_window_duration"] = float(cfg["min_window_duration"])
        if cfg.get("energy_threshold_db") is not None:
            kwargs["energy_threshold_db"] = float(cfg["energy_threshold_db"])
        if cfg.get("norm_filter_sigma") is not None:
            kwargs["norm_filter_sigma"] = float(cfg["norm_filter_sigma"])
        if cfg.get("coherence_threshold") is not None:
            kwargs["coherence_threshold"] = float(cfg["coherence_threshold"])
        if cfg.get("min_vote_similarity") is not None:
            kwargs["min_vote_similarity"] = float(cfg["min_vote_similarity"])
        if cfg.get("min_utterance_duration_cluster") is not None:
            kwargs["min_utterance_duration_cluster"] = float(
                cfg["min_utterance_duration_cluster"]
            )
        if cfg.get("min_utterance_coherence_cluster") is not None:
            kwargs["min_utterance_coherence_cluster"] = float(
                cfg["min_utterance_coherence_cluster"]
            )
        if cfg.get("hdbscan_min_cluster_size") is not None:
            kwargs["hdbscan_min_cluster_size"] = int(cfg["hdbscan_min_cluster_size"])
        if cfg.get("umap_n_neighbors") is not None:
            kwargs["umap_n_neighbors"] = int(cfg["umap_n_neighbors"])
        if cfg.get("umap_min_dist") is not None:
            kwargs["umap_min_dist"] = float(cfg["umap_min_dist"])
        if cfg.get("umap_n_components") is not None:
            kwargs["umap_n_components"] = int(cfg["umap_n_components"])
        if cfg.get("utterance_aggregation") is not None:
            kwargs["utterance_aggregation"] = str(cfg["utterance_aggregation"])

        return SpeakerAttributor(SpeakerAttributorConfig(**kwargs))

    def _build_emotion_analyzer(self) -> Any:
        from yorishiro.audio.emotion_analysis import (
            EmotionAnalyzer,
            EmotionAnalyzerConfig,
        )

        cfg = self._project.step_config("film.audio.emotion")
        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["emotion_backend"] = cfg["backend"]
        if cfg.get("model") is not None:
            kwargs["emotion_model"] = cfg["model"]
        return EmotionAnalyzer(EmotionAnalyzerConfig(**kwargs))

    def _build_sound_event_detector(self) -> Any:
        from yorishiro.audio.sound_event_detector import (
            SoundEventDetector,
            SoundEventDetectorConfig,
        )

        cfg = self._project.step_config("film.audio.sound_events")
        kwargs: dict[str, Any] = {}
        if cfg.get("backend") is not None:
            kwargs["backend"] = cfg["backend"]
        if cfg.get("model") is not None:
            kwargs["model"] = cfg["model"]
        return SoundEventDetector(SoundEventDetectorConfig(**kwargs))

    def _build_music_analyzer(self) -> Any:
        from yorishiro.audio.music_analyzer import MusicAnalyzer, MusicAnalyzerConfig

        cfg = self._project.step_config("film.audio.music")
        separation = cfg.get("separation", {})
        analysis = cfg.get("analysis", {})
        if not isinstance(separation, dict):
            separation = {}
        if not isinstance(analysis, dict):
            analysis = {}

        def _parse_bool(v: object) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
            if v is None:
                return False
            return bool(v)

        kwargs: dict[str, Any] = {}
        if separation.get("backend") is not None:
            kwargs["separation_backend"] = str(separation["backend"])
        if separation.get("model") is not None:
            kwargs["separation_model"] = str(separation["model"])
        if analysis.get("backend") is not None:
            kwargs["analysis_backend"] = str(analysis["backend"])
        if cfg.get("detect_lyrics") is not None:
            kwargs["detect_lyrics"] = _parse_bool(cfg["detect_lyrics"])
        return MusicAnalyzer(MusicAnalyzerConfig(**kwargs))

    def _build_audio_separator(self) -> Any:
        from yorishiro.audio.separator import AudioSeparator, AudioSeparatorConfig

        cfg = self._project.step_config("film.audio.separate")
        kwargs: dict[str, Any] = {}
        for key, dest, conv in [
            ("backend", "backend", str),
            ("model", "model", str),
            ("device", "device", str),
            ("sample_rate", "sample_rate", int),
            ("processing_chunk_seconds", "processing_chunk_seconds", float),
            ("processing_overlap_seconds", "processing_overlap_seconds", float),
        ]:
            if cfg.get(key) is not None:
                kwargs[dest] = conv(cfg[key])
        return AudioSeparator(AudioSeparatorConfig(**kwargs))
