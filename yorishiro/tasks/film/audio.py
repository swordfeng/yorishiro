"""film.audio.* steps: stem separation, speech pipeline stages, non-speech analysis."""

from __future__ import annotations

from pathlib import Path

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


# ---------------------------------------------------------------------------
# Tasks (one per step — no key needed)
# ---------------------------------------------------------------------------


class FilmAudioSeparateTask(Task):
    """Separate voice / non-voice stems via Demucs → voice.flac + nonvoice.flac."""

    def __init__(
        self, video_path: Path, output_dir: Path, runtime: StepRuntime
    ) -> None:
        self._video_path = video_path
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [self._video_path]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac", self._output_dir / "nonvoice.flac"]

    def completion_marker(self) -> Path:
        return self._output_dir / "nonvoice.flac"

    def _run(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._runtime.instance()
        print("[film.audio.separate] Separating voice / non-voice stems ...")
        separator.separate(self._video_path, self._output_dir, force=True)
        separator.release_models()
        print("[film.audio.separate] Done.")


class FilmAudioVADTask(Task):
    """Run Voice Activity Detection on voice.flac → vad.json."""

    def __init__(self, output_dir: Path, runtime: StepRuntime) -> None:
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "nonvoice.flac",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "vad.json"]

    def _run(self) -> None:
        print("[film.audio.vad] Running VAD on voice stem ...")
        vad_runner = self._runtime.instance()
        vad_runner.run(
            self._output_dir / "voice.flac",
            self._output_dir,
            nonvoice_path=self._output_dir / "nonvoice.flac",
        )


class FilmAudioDiarizeTask(Task):
    """Run speaker diarization on voice.flac → diarization.json."""

    def __init__(self, output_dir: Path, runtime: StepRuntime) -> None:
        self._output_dir = output_dir
        self._runtime = runtime
        self._force = False

    def input_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "diarization.json"]

    def run(self, force: bool = False) -> bool:
        self._force = force
        return super().run(force=force)

    def _run(self) -> None:
        print("[film.audio.diarize] Running speaker diarization on voice stem ...")
        diarizer = self._runtime.instance()
        diarizer.run(
            self._output_dir / "voice.flac", self._output_dir, force=self._force
        )


class FilmAudioSTTTask(Task):
    """Run speech-to-text using vad.json → stt.json, or from subtitle file."""

    def __init__(
        self,
        output_dir: Path,
        language: str | None,
        runtime: StepRuntime,
        subtitle_path: Path | None = None,
        subtitle_track: int | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._language = language
        self._runtime = runtime
        self._subtitle_path = subtitle_path
        self._subtitle_track = subtitle_track
        self._force = False

    def input_paths(self) -> list[Path]:
        if self._subtitle_path is not None:
            return [self._subtitle_path]
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "vad.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "stt.json"]

    def run(self, force: bool = False) -> bool:
        self._force = force
        return super().run(force=force)

    def _run(self) -> None:
        print("[film.audio.stt] Running speech-to-text ...")
        transcriber = self._runtime.instance()
        try:
            if self._subtitle_path is not None:
                transcriber.run_from_subtitle_path(
                    self._subtitle_path,
                    self._output_dir / "stt.json",
                    self._language,
                    force=self._force,
                    track=self._subtitle_track,
                )
            else:
                transcriber.run(
                    self._output_dir / "voice.flac",
                    self._output_dir,
                    self._language,
                    force=self._force,
                )
        finally:
            transcriber.release_models()


class FilmAudioSpeakersTask(Task):
    """Run speaker attribution on stt.json → speaker_attribution.json + speaker bank."""

    def __init__(self, output_dir: Path, runtime: StepRuntime) -> None:
        self._output_dir = output_dir
        self._runtime = runtime
        self._force = False

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "stt.json",
        ]

    def output_paths(self) -> list[Path]:
        return [
            self._output_dir / "speaker_attribution.json",
            self._output_dir / "speaker_bank.json",
            self._output_dir / "speaker_embeddings.pkl",
            self._output_dir / "speaker_embedding_cache.npz",
        ]

    def run(self, force: bool = False) -> bool:
        self._force = force
        return super().run(force=force)

    def _run(self) -> None:
        print("[film.audio.speakers] Running speaker attribution ...")
        attributor = self._runtime.instance()
        try:
            attributor.run(
                self._output_dir / "voice.flac", self._output_dir, force=self._force
            )
        finally:
            attributor.release_models()


class FilmAudioEmotionTask(Task):
    """Run emotion + prosody analysis on stt.json + speaker_attribution.json → transcript.json."""

    def __init__(self, output_dir: Path, runtime: StepRuntime) -> None:
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "stt.json",
            self._output_dir / "speaker_attribution.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript.json"]

    def _run(self) -> None:
        print("[film.audio.emotion] Running emotion analysis ...")
        emotion_analyzer = self._runtime.instance()
        try:
            emotion_analyzer.run(self._output_dir / "voice.flac", self._output_dir)
        finally:
            emotion_analyzer.release_models()
        print("[film.audio.emotion] Done.")


class FilmAudioSoundEventsTask(Task):
    """Run stem-aware sound event detection on voice/nonvoice flac → sound_events.json."""

    def __init__(self, output_dir: Path, runtime: StepRuntime) -> None:
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        # transcript.json is used to infer spoken-content tail for better filtering,
        # but sound detection still works if transcript is unavailable.
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "nonvoice.flac",
            self._output_dir / "transcript.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "sound_events.json"]

    def _run(self) -> None:
        voice_path = self._output_dir / "voice.flac"
        nonvoice_path = self._output_dir / "nonvoice.flac"
        transcript_path = self._output_dir / "transcript.json"

        sound_detector = self._runtime.instance()
        print(
            "[film.audio.sound_events] Running sound event detection on voice + non-voice stems ..."
        )
        try:
            sound_detector.detect(
                voice_path,
                nonvoice_path,
                self._output_dir,
                transcript_path=transcript_path,
                force=True,
            )
        finally:
            sound_detector.release_models()
        print("[film.audio.sound_events] Done.")


class FilmAudioMusicTask(Task):
    """Run music analysis on nonvoice.flac → music_analysis.json."""

    def __init__(
        self, video_path: Path, output_dir: Path, runtime: StepRuntime
    ) -> None:
        self._video_path = video_path
        self._output_dir = output_dir
        self._runtime = runtime

    def input_paths(self) -> list[Path]:
        return [
            self._video_path,
            self._output_dir / "nonvoice.flac",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "music_analysis.json"]

    def _run(self) -> None:
        nonvoice_path = self._output_dir / "nonvoice.flac"

        music_analyzer = self._runtime.instance()
        print("[film.audio.music] Running music analysis on non-voice stem ...")
        try:
            music_analyzer.analyze(
                self._video_path,
                nonvoice_path,
                self._output_dir,
                force=True,
                nonvoice_path=nonvoice_path,
            )
        finally:
            music_analyzer.release_models()
        print("[film.audio.music] Done.")


# ---------------------------------------------------------------------------
# Steps (one per stage)
# ---------------------------------------------------------------------------


class FilmAudioSeparateStep(Step):
    step_id = "film.audio.separate"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioSeparateTask(
                self._project.get_source_path(self._source_id),
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioVADStep(Step):
    step_id = "film.audio.vad"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioVADTask(
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioDiarizeStep(Step):
    step_id = "film.audio.diarize"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioDiarizeTask(
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioSTTStep(Step):
    step_id = "film.audio.stt"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        source = self._project.get_source(self._source_id)
        language = source.config.get("language") if source else None
        step_config = self._project.step_config(self.step_id)
        stt_backend = step_config.get("backend", "faster-whisper")

        # Check if using subtitle backend
        if stt_backend == "subtitles":
            subtitle_source = step_config.get("subtitle_source")
            if subtitle_source:
                subtitle_path, subtitle_config = self._project.get_subtitle_source(
                    subtitle_source
                )
            else:
                # Use the film source itself as the subtitle source
                subtitle_path = self._project.get_source_path(self._source_id)
                subtitle_config = source.config if source else {}
            subtitle_track = None
            if "subtitle_track" in step_config:
                subtitle_track = int(step_config["subtitle_track"])
            elif "track" in subtitle_config:
                subtitle_track = int(subtitle_config["track"])
            return [
                FilmAudioSTTTask(
                    self._project.step_dir(self._source_id, "audio"),
                    language,
                    self._registry.for_step(self.step_id),
                    subtitle_path=subtitle_path,
                    subtitle_track=subtitle_track,
                )
            ]

        return [
            FilmAudioSTTTask(
                self._project.step_dir(self._source_id, "audio"),
                language,
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioSpeakersStep(Step):
    step_id = "film.audio.speakers"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioSpeakersTask(
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioEmotionStep(Step):
    step_id = "film.audio.emotion"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioEmotionTask(
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioSoundEventsStep(Step):
    step_id = "film.audio.sound_events"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioSoundEventsTask(
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]


class FilmAudioMusicStep(Step):
    step_id = "film.audio.music"

    def __init__(
        self, project: Project, source_id: str, registry: ModelRegistry
    ) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [
            FilmAudioMusicTask(
                self._project.get_source_path(self._source_id),
                self._project.step_dir(self._source_id, "audio"),
                self._registry.for_step(self.step_id),
            )
        ]
