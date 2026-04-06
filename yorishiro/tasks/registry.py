"""ModelRegistry: lazy-loads local ML models and resolves cloud LLM configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yorishiro.audio.music_analyzer import MusicAnalyzer
    from yorishiro.audio.separator import AudioSeparator
    from yorishiro.audio.sound_event_detector import SoundEventDetector
    from yorishiro.audio.speaker_bank import SpeakerBankManager
    from yorishiro.audio.speech_pipeline import SpeechPipeline
    from yorishiro.project import ModelConfig, Project
    from yorishiro.video.keyframe_extractor import KeyFrameExtractor
    from yorishiro.video.shot_detector import ShotDetector


class ModelRegistry:
    """Resolves step config and lazily instantiates local ML models.

    Config resolution: step-level fields → referenced model definition.
    Local model instances are cached by model name (shared across tasks).
    """

    def __init__(self, project: Project) -> None:
        self._project = project
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def step_config(self, step_id: str) -> dict[str, Any]:
        """Merged config for a step (step overrides → model definition)."""
        return self._project.step_config(step_id)

    def cloud_config(self, step_id: str) -> ModelConfig:
        """Resolved ModelConfig for a cloud LLM step."""
        return self._project.resolved_model_config(step_id)

    def local_config(self, step_id: str) -> dict[str, Any]:
        """Merged config dict for a local ML step."""
        return self._project.local_model_config(step_id)

    # ------------------------------------------------------------------
    # Local model instance getters (lazy, cached by model name)
    # ------------------------------------------------------------------

    def _model_name_for_step(self, step_id: str) -> str | None:
        return self._project.steps.get(step_id, {}).get("model")

    def get_shot_detector(self) -> ShotDetector:
        from yorishiro.video.shot_detector import ShotDetector, ShotDetectorConfig

        key = self._model_name_for_step("film.shots") or "__shot_detector__"
        if key not in self._cache:
            cfg = self.local_config("film.shots")
            self._cache[key] = ShotDetector(ShotDetectorConfig(
                detector=cfg.get("backend", "adaptive"),
                threshold=cfg.get("threshold", 4.0),
                min_content_val=cfg.get("min_content_val", 15.0),
            ))
        return self._cache[key]

    def get_keyframe_extractor(self) -> KeyFrameExtractor:
        from yorishiro.video.keyframe_extractor import KeyFrameExtractor, KeyFrameExtractorConfig

        key = self._model_name_for_step("film.frames") or "__keyframe__"
        if key not in self._cache:
            cfg = self.local_config("film.frames")
            self._cache[key] = KeyFrameExtractor(KeyFrameExtractorConfig(
                backend=cfg.get("backend", "clip"),
                model=cfg.get("clip_model", "ViT-B/32"),
                min_frames_per_shot=cfg.get("min_frames_per_shot", 2),
                max_frames_per_shot=cfg.get("max_frames_per_shot", 8),
                max_frames_per_scene=cfg.get("max_frames_per_scene", 8),
                output_format=cfg.get("output_format", "avif"),
                output_quality=cfg.get("output_quality", 85),
            ))
        return self._cache[key]

    def get_speech_pipeline(self) -> SpeechPipeline:
        from yorishiro.audio.speech_pipeline import SpeechPipeline, SpeechPipelineConfig

        key = self._model_name_for_step("film.audio.stt") or "__speech__"
        if key not in self._cache:
            cfg = self.local_config("film.audio.stt")
            diar_step_cfg = self._project.steps.get("film.audio.diarize", {})
            diar_name = diar_step_cfg.get("diarization_model")
            diar_cfg = dict(self._project.models.get(diar_name, {})) if diar_name else {}
            emotion_step_cfg = self._project.steps.get("film.audio.emotion", {})
            self._cache[key] = SpeechPipeline(SpeechPipelineConfig(
                stt_backend=cfg.get("backend", "faster-whisper"),
                stt_model=cfg.get("model", "large-v3"),
                stt_cpu_threads=int(cfg.get("cpu_threads", 0)),
                stt_num_workers=int(cfg.get("num_workers", 1)),
                diarization_backend=diar_cfg.get("backend", "pyannote"),
                diarization_model=diar_cfg.get("model", "pyannote/speaker-diarization-3.1"),
                diarization_batch_size=int(diar_cfg.get("batch_size", 32)),
                hf_token_env=diar_cfg.get("hf_token_env", "YORISHIRO_HF_TOKEN"),
                emotion_model=emotion_step_cfg.get("model", "emotion2vec/emotion2vec_plus_base"),
            ))
        return self._cache[key]

    def get_speaker_bank_manager(self) -> SpeakerBankManager:
        from yorishiro.audio.speaker_bank import SpeakerBankManager, SpeakerBankManagerConfig

        key = "__speaker_bank__"
        if key not in self._cache:
            diar_name = self._project.steps.get("film.audio.diarize", {}).get("diarization_model", "diarization")
            diar_cfg = dict(self._project.models.get(diar_name, {})) if diar_name else {}
            self._cache[key] = SpeakerBankManager(SpeakerBankManagerConfig(
                hf_token_env=diar_cfg.get("hf_token_env", "YORISHIRO_HF_TOKEN"),
            ))
        return self._cache[key]

    def get_sound_event_detector(self) -> SoundEventDetector:
        from yorishiro.audio.sound_event_detector import SoundEventDetector, SoundEventDetectorConfig

        key = "__sound_events__"
        if key not in self._cache:
            self._cache[key] = SoundEventDetector(SoundEventDetectorConfig())
        return self._cache[key]

    def get_music_analyzer(self) -> MusicAnalyzer:
        from yorishiro.audio.music_analyzer import MusicAnalyzer, MusicAnalyzerConfig

        key = "__music__"
        if key not in self._cache:
            self._cache[key] = MusicAnalyzer(MusicAnalyzerConfig())
        return self._cache[key]

    def get_audio_separator(self) -> AudioSeparator:
        from yorishiro.audio.separator import AudioSeparator, AudioSeparatorConfig

        key = self._model_name_for_step("film.audio.separate") or "__separator__"
        if key not in self._cache:
            cfg = self._project.steps.get("film.audio.separate", {})
            sep_name = cfg.get("separator_model")
            sep_cfg = dict(self._project.models.get(sep_name, {})) if sep_name else {}
            self._cache[key] = AudioSeparator(AudioSeparatorConfig(
                backend=sep_cfg.get("backend", "demucs"),
                model=sep_cfg.get("model", "htdemucs"),
                device=sep_cfg.get("device", "auto"),
            ))
        return self._cache[key]
