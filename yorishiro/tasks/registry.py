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

    def __init__(self, registry: ModelRegistry, step_id: str, spec: _RuntimeSpec) -> None:
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
                "film.frames", "model", "backend", "clip_model", fallback="keyframe-extractor"
            ),
        ),
        "film.audio.separate": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_audio_separator(),
            cache_key_builder=lambda registry: registry._audio_separator_cache_key(),
        ),
        "film.audio.vad": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_speech_pipeline(),
            cache_key_builder=lambda registry: registry._speech_pipeline_cache_key(),
        ),
        "film.audio.diarize": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_speech_pipeline(),
            cache_key_builder=lambda registry: registry._speech_pipeline_cache_key(),
        ),
        "film.audio.stt": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_speech_pipeline(),
            cache_key_builder=lambda registry: registry._speech_pipeline_cache_key(),
        ),
        "film.audio.emotion": _RuntimeSpec(
            kind="instance",
            instance_builder=lambda registry: registry._build_speech_pipeline(),
            cache_key_builder=lambda registry: registry._speech_pipeline_cache_key(),
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
            cache_key_builder=lambda registry: registry._cache_key_for_step(
                "film.audio.music", fallback="music-analyzer"
            ),
        ),
    }

    def __init__(self, project: Project) -> None:
        self._project = project
        self._cache: dict[str, Any] = {}

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
            raise TypeError(f"Step '{step_id}' is missing instance runtime builder configuration")
        cache_key = spec.cache_key_builder(self)
        if cache_key not in self._cache:
            self._cache[cache_key] = spec.instance_builder(self)
        return self._cache[cache_key]

    def _cache_key_for_step(self, step_id: str, *fields: str, fallback: str) -> str:
        cfg = self._project.step_config(step_id)
        values = [str(cfg[field]) for field in fields if field in cfg and cfg[field] is not None]
        identity = "|".join(values) if values else fallback
        return f"{step_id}::{identity}"

    def _audio_separator_cache_key(self) -> str:
        step_cfg = self._project.steps.get("film.audio.separate", {})
        sep_name = step_cfg.get("separator_model")
        if sep_name:
            model_cfg = self._project.models.get(sep_name, {})
            identity = "|".join(
                str(model_cfg.get(field, ""))
                for field in ("backend", "model", "device")
            )
            return f"film.audio.separate::{sep_name}::{identity}"
        return self._cache_key_for_step("film.audio.separate", fallback="audio-separator")

    def _speech_pipeline_cache_key(self) -> str:
        stt_cfg = self._project.step_config("film.audio.stt")
        diar_step_cfg = self._project.steps.get("film.audio.diarize", {})
        emotion_step_cfg = self._project.steps.get("film.audio.emotion", {})
        diar_name = diar_step_cfg.get("diarization_model")
        diar_cfg = dict(self._project.models.get(diar_name, {})) if diar_name else {}
        parts = [
            "speech-pipeline",
            str(stt_cfg.get("backend", "")),
            str(stt_cfg.get("model", "")),
            str(stt_cfg.get("cpu_threads", "")),
            str(stt_cfg.get("num_workers", "")),
            str(stt_cfg.get("word_timestamps", "")),
            str(stt_cfg.get("vad_filter", "")),
            str(stt_cfg.get("vad_min_silence_duration_ms", "")),
            str(diar_cfg.get("backend", "")),
            str(diar_cfg.get("model", "")),
            str(diar_cfg.get("batch_size", "")),
            str(diar_cfg.get("hf_token_env", "")),
            str(emotion_step_cfg.get("model", "")),
        ]
        return "film.audio.shared::" + "|".join(parts)

    def _build_shot_detector(self) -> Any:
        from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig

        cfg = self._project.step_config("film.shots")
        return ShotDetector(ShotDetectorConfig(
            detector=cfg.get("backend", "adaptive"),
            threshold=cfg.get("threshold", 4.0),
            min_content_val=cfg.get("min_content_val", 15.0),
        ))

    def _build_keyframe_extractor(self) -> Any:
        from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig

        cfg = self._project.step_config("film.frames")
        return KeyFrameExtractor(KeyFrameExtractorConfig(
            backend=cfg.get("backend", "clip"),
            model=cfg.get("clip_model", "ViT-B/32"),
            min_frames_per_shot=cfg.get("min_frames_per_shot", 2),
            max_frames_per_shot=cfg.get("max_frames_per_shot", 8),
            max_frames_per_scene=cfg.get("max_frames_per_scene", 8),
            output_format=cfg.get("output_format", "avif"),
            output_quality=cfg.get("output_quality", 85),
        ))

    def _build_speech_pipeline(self) -> Any:
        from yorishiro.audio.speech_pipeline import SpeechPipeline, SpeechPipelineConfig

        cfg = self._project.step_config("film.audio.stt")
        diar_step_cfg = self._project.steps.get("film.audio.diarize", {})
        diar_name = diar_step_cfg.get("diarization_model")
        diar_cfg = dict(self._project.models.get(diar_name, {})) if diar_name else {}
        emotion_step_cfg = self._project.steps.get("film.audio.emotion", {})
        return SpeechPipeline(SpeechPipelineConfig(
            stt_backend=cfg.get("backend", "faster-whisper"),
            stt_model=cfg.get("model", "large-v3"),
            stt_cpu_threads=int(cfg.get("cpu_threads", 0)),
            stt_num_workers=int(cfg.get("num_workers", 1)),
            stt_word_timestamps=bool(cfg.get("word_timestamps", False)),
            stt_vad_filter=bool(cfg.get("vad_filter", False)),
            stt_vad_min_silence_duration_ms=int(cfg.get("vad_min_silence_duration_ms", 500)),
            diarization_backend=diar_cfg.get("backend", "pyannote"),
            diarization_model=diar_cfg.get("model", "pyannote/speaker-diarization-3.1"),
            diarization_batch_size=int(diar_cfg.get("batch_size", 32)),
            hf_token_env=diar_cfg.get("hf_token_env", "YORISHIRO_HF_TOKEN"),
            emotion_model=emotion_step_cfg.get("model", "emotion2vec/emotion2vec_plus_base"),
        ))

    def _build_sound_event_detector(self) -> Any:
        from yorishiro.audio.sound_event_detector import SoundEventDetector, SoundEventDetectorConfig

        return SoundEventDetector(SoundEventDetectorConfig())

    def _build_music_analyzer(self) -> Any:
        from yorishiro.audio.music_analyzer import MusicAnalyzer, MusicAnalyzerConfig

        return MusicAnalyzer(MusicAnalyzerConfig())

    def _build_audio_separator(self) -> Any:
        from yorishiro.audio.separator import AudioSeparator, AudioSeparatorConfig

        cfg = self._project.steps.get("film.audio.separate", {})
        sep_name = cfg.get("separator_model")
        sep_cfg = dict(self._project.models.get(sep_name, {})) if sep_name else {}
        return AudioSeparator(AudioSeparatorConfig(
            backend=sep_cfg.get("backend", "demucs"),
            model=sep_cfg.get("model", "htdemucs"),
            device=sep_cfg.get("device", "auto"),
        ))
