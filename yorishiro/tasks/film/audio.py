"""film.audio.* steps: extract audio, separate stems, speech pipeline, analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Tasks (one per step — no key needed)
# ---------------------------------------------------------------------------

class FilmAudioExtractTask(Task):
    """Extract raw audio from video → audio.flac (16kHz mono)."""

    def __init__(self, video_path: Path, output_dir: Path) -> None:
        self._video_path = video_path
        self._output_dir = output_dir

    def input_paths(self) -> list[Path]:
        return [self._video_path]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "audio.flac"]

    def _run(self) -> None:
        import av
        import numpy as np
        import soundfile as sf
        from av.audio.frame import AudioFrame

        self._output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._output_dir / "audio.flac"

        print(f"[film.audio.extract] Extracting audio from {self._video_path.name} ...")
        input_container = av.open(str(self._video_path))
        audio_stream = next(
            (s for s in input_container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {self._video_path}")

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in input_container.decode(audio_stream):
            assert isinstance(frame, AudioFrame)
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()  # (1, samples) int16
                chunks.append(arr)
        input_container.close()

        if not chunks:
            raise ValueError(f"No audio frames decoded from {self._video_path}")

        audio = np.concatenate(chunks, axis=1)[0].astype("float32") / 32768.0
        sf.write(str(audio_path), audio, 16000)
        print(f"[film.audio.extract] Done: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")


class FilmAudioSeparateTask(Task):
    """Separate voice / non-voice stems via Demucs → voice.flac + nonvoice.flac."""

    def __init__(self, video_path: Path, output_dir: Path, registry: ModelRegistry) -> None:
        self._video_path = video_path
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [self._video_path]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac", self._output_dir / "nonvoice.flac"]

    def completion_marker(self) -> Path:
        return self._output_dir / "nonvoice.flac"

    def _run(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._registry.get_audio_separator()
        print("[film.audio.separate] Separating voice / non-voice stems ...")
        separator.separate(self._video_path, self._output_dir, force=True)
        print("[film.audio.separate] Done.")


class FilmAudioVADTask(Task):
    """Run Voice Activity Detection on voice.flac → vad.json."""

    def __init__(self, output_dir: Path, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "vad.json"]

    def _run(self) -> None:
        print("[film.audio.vad] Running VAD on voice stem ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_vad(self._output_dir / "voice.flac", self._output_dir)


class FilmAudioDiarizeTask(Task):
    """Run speaker diarization on voice.flac → diarization.json."""

    def __init__(self, output_dir: Path, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "diarization.json"]

    def _run(self) -> None:
        print("[film.audio.diarize] Running speaker diarization on voice stem ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_diarization(self._output_dir / "voice.flac", self._output_dir)


class FilmAudioSTTTask(Task):
    """Run speech-to-text using vad.json + diarization.json → transcript_raw.json."""

    def __init__(self, output_dir: Path, language: str | None, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._language = language
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "vad.json",
            self._output_dir / "diarization.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript_raw.json"]

    def _run(self) -> None:
        print("[film.audio.stt] Running speech-to-text ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_stt(self._output_dir / "voice.flac", self._output_dir, self._language)


class FilmAudioEmotionTask(Task):
    """Run emotion analysis on transcript_raw.json → transcript.json + speaker_bank.json."""

    def __init__(self, output_dir: Path, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "transcript_raw.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript.json", self._output_dir / "speaker_bank.json"]

    def _run(self) -> None:
        print("[film.audio.emotion] Running emotion analysis ...")
        pipeline = self._registry.get_speech_pipeline()
        transcript = pipeline.run_emotion(self._output_dir / "voice.flac", self._output_dir)

        speaker_bank = self._registry.get_speaker_bank_manager()
        print("[film.audio.emotion] Resolving speaker IDs ...")
        speaker_bank.load(self._output_dir)
        self._resolve_speaker_ids(transcript, self._output_dir / "voice.flac", speaker_bank)
        speaker_bank.save(self._output_dir)
        print("[film.audio.emotion] Done.")

    def _resolve_speaker_ids(
        self,
        transcript: Any,
        audio_path: Path,
        speaker_bank: Any,
    ) -> None:
        if not transcript or not transcript.entries:
            return
        if all(e.speaker_global.startswith("SPKR_") for e in transcript.entries):
            return

        local_speakers: dict[str, Any] = {}
        for entry in transcript.entries:
            if entry.speaker_global not in local_speakers:
                local_speakers[entry.speaker_global] = entry

        local_to_global: dict[str, str] = {}
        for local_id, rep_entry in local_speakers.items():
            embedding = None
            if audio_path.exists():
                embedding = speaker_bank.extract_speaker_embedding(
                    audio_path, rep_entry.start, rep_entry.end
                )
            global_id = speaker_bank.assign_global_speaker_id(local_id, embedding, rep_entry.start)
            local_to_global[local_id] = global_id
            print(f"    {local_id} → {global_id}")

        for entry in transcript.entries:
            entry.speaker_global = local_to_global.get(entry.speaker_global, entry.speaker_global)

        cache_file = self._output_dir / "transcript.json"
        cache_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


class FilmAudioAnalysisTask(Task):
    """Run sound event detection + music analysis on nonvoice.flac."""

    def __init__(self, video_path: Path, output_dir: Path, registry: ModelRegistry) -> None:
        self._video_path = video_path
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "nonvoice.flac",
            self._output_dir / "transcript.json",
        ]

    def output_paths(self) -> list[Path]:
        return [
            self._output_dir / "sound_events.json",
            self._output_dir / "music_analysis.json",
        ]

    def completion_marker(self) -> Path:
        return self._output_dir / "music_analysis.json"

    def _run(self) -> None:
        import json

        nonvoice_path = self._output_dir / "nonvoice.flac"

        transcript_end = 0.0
        transcript_path = self._output_dir / "transcript.json"
        if transcript_path.exists():
            try:
                data = json.loads(transcript_path.read_text(encoding="utf-8"))
                entries = data.get("entries", [])
                if entries:
                    transcript_end = entries[-1].get("end", 0.0)
            except Exception:
                pass

        sound_detector = self._registry.get_sound_event_detector()
        print("[film.audio.analysis] Running sound event detection on non-voice stem ...")
        sound_detector.detect(nonvoice_path, self._output_dir, transcript_end=transcript_end, force=True)

        music_analyzer = self._registry.get_music_analyzer()
        print("[film.audio.analysis] Running music analysis on non-voice stem ...")
        music_analyzer.analyze(
            self._video_path,
            nonvoice_path,
            self._output_dir,
            force=True,
            nonvoice_path=nonvoice_path,
        )
        print("[film.audio.analysis] Done.")


# ---------------------------------------------------------------------------
# Steps (one per stage)
# ---------------------------------------------------------------------------

class FilmAudioExtractStep(Step):
    step_id = "film.audio.extract"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id

    def tasks(self) -> list[Task]:
        return [FilmAudioExtractTask(
            self._project.get_source_path(self._source_id),
            self._project.step_dir(self._source_id, "audio"),
        )]


class FilmAudioSeparateStep(Step):
    step_id = "film.audio.separate"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioSeparateTask(
            self._project.get_source_path(self._source_id),
            self._project.step_dir(self._source_id, "audio"),
            self._registry,
        )]


class FilmAudioVADStep(Step):
    step_id = "film.audio.vad"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioVADTask(
            self._project.step_dir(self._source_id, "audio"),
            self._registry,
        )]


class FilmAudioDiarizeStep(Step):
    step_id = "film.audio.diarize"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioDiarizeTask(
            self._project.step_dir(self._source_id, "audio"),
            self._registry,
        )]


class FilmAudioSTTStep(Step):
    step_id = "film.audio.stt"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        source = self._project.get_source(self._source_id)
        language = source.config.get("language") if source else None
        return [FilmAudioSTTTask(
            self._project.step_dir(self._source_id, "audio"),
            language,
            self._registry,
        )]


class FilmAudioEmotionStep(Step):
    step_id = "film.audio.emotion"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioEmotionTask(
            self._project.step_dir(self._source_id, "audio"),
            self._registry,
        )]


class FilmAudioAnalysisStep(Step):
    step_id = "film.audio.analysis"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioAnalysisTask(
            self._project.get_source_path(self._source_id),
            self._project.step_dir(self._source_id, "audio"),
            self._registry,
        )]
