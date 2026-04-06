"""film.audio.* steps: extract audio, separate stems, speech pipeline, analysis."""

from __future__ import annotations

import json
from pathlib import Path

from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.project import Project
from yorishiro.tasks.base import Step, Task
from yorishiro.tasks.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Tasks (one per step — no key needed)
# ---------------------------------------------------------------------------


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
        self._force = False

    def input_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "diarization.json"]

    def run(self, force: bool = False) -> None:
        self._force = force
        super().run(force=force)

    def _run(self) -> None:
        print("[film.audio.diarize] Running speaker diarization on voice stem ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_diarization(self._output_dir / "voice.flac", self._output_dir, force=self._force)


class FilmAudioSTTTask(Task):
    """Run speech-to-text using vad.json + diarization.json → transcript_raw.json."""

    def __init__(self, output_dir: Path, language: str | None, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._language = language
        self._registry = registry
        self._force = False

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "vad.json",
            self._output_dir / "diarization.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript_raw.json"]

    def run(self, force: bool = False) -> None:
        self._force = force
        super().run(force=force)

    def _run(self) -> None:
        print("[film.audio.stt] Running speech-to-text ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_stt(self._output_dir / "voice.flac", self._output_dir, self._language, force=self._force)


class FilmAudioEmotionTask(Task):
    """Run emotion + prosody analysis on transcript_raw.json → transcript.json."""

    def __init__(self, output_dir: Path, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [
            self._output_dir / "voice.flac",
            self._output_dir / "transcript_raw.json",
        ]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript.json"]

    def _run(self) -> None:
        print("[film.audio.emotion] Running emotion analysis ...")
        pipeline = self._registry.get_speech_pipeline()
        pipeline.run_emotion(self._output_dir / "voice.flac", self._output_dir)
        print("[film.audio.emotion] Done.")


class FilmAudioSpeakerTask(Task):
    """Map local SPEAKER_XX IDs to global SPKR_XXX IDs → transcript.json + speaker_bank.json."""

    def __init__(self, output_dir: Path, registry: ModelRegistry) -> None:
        self._output_dir = output_dir
        self._registry = registry

    def input_paths(self) -> list[Path]:
        return [self._output_dir / "voice.flac", self._output_dir / "transcript.json"]

    def output_paths(self) -> list[Path]:
        return [self._output_dir / "transcript.json", self._output_dir / "speaker_bank.json"]

    def completion_marker(self) -> Path:
        # transcript.json is both input and output; use speaker_bank.json as the marker
        # (written last in _run, so its mtime > transcript.json → staleness check is stable)
        return self._output_dir / "speaker_bank.json"

    def _run(self) -> None:
        from yorishiro.audio.speaker_bank import SpeakerBankManager

        print("[film.audio.speaker] Resolving speaker IDs ...")
        transcript = Transcript(**json.loads(
            (self._output_dir / "transcript.json").read_text(encoding="utf-8")
        ))

        # Collect unique speakers; pick longest entry as representative
        speaker_rep: dict[str, TranscriptEntry] = {}
        speaker_first: dict[str, float] = {}
        for entry in transcript.entries:
            spk = entry.speaker_global
            if spk not in speaker_first:
                speaker_first[spk] = entry.start
            dur = entry.end - entry.start
            if spk not in speaker_rep or dur > (speaker_rep[spk].end - speaker_rep[spk].start):
                speaker_rep[spk] = entry

        # Ordered by first appearance → deterministic SPKR_001, SPKR_002, ...
        speakers_ordered = sorted(speaker_rep, key=lambda s: speaker_first[s])

        # Fresh bank — no load; fixes accumulation bug on force re-runs
        bank = SpeakerBankManager(self._registry.get_speaker_bank_manager().config)

        voice_path = self._output_dir / "voice.flac"
        local_to_global: dict[str, str] = {}
        for local_id in speakers_ordered:
            rep = speaker_rep[local_id]
            embedding = bank.extract_speaker_embedding(voice_path, rep.start, rep.end)
            global_id = bank.assign_global_speaker_id(local_id, embedding, rep.start)
            local_to_global[local_id] = global_id
            print(f"    {local_id} → {global_id}")

        for entry in transcript.entries:
            entry.speaker_global = local_to_global.get(entry.speaker_global, entry.speaker_global)

        (self._output_dir / "transcript.json").write_text(
            transcript.model_dump_json(indent=2), encoding="utf-8"
        )
        bank.save(self._output_dir)
        print(f"[film.audio.speaker] Done — {len(local_to_global)} speaker(s) mapped.")


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


class FilmAudioSpeakerStep(Step):
    step_id = "film.audio.speaker"

    def __init__(self, project: Project, source_id: str, registry: ModelRegistry) -> None:
        self._project = project
        self._source_id = source_id
        self._registry = registry

    def tasks(self) -> list[Task]:
        return [FilmAudioSpeakerTask(
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
