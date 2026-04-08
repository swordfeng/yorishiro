from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from yorishiro.audio.diarization import Diarizer, DiarizerConfig
from yorishiro.audio.emotion_analysis import EmotionAnalyzer
from yorishiro.audio.transcription import Transcriber, TranscriberConfig
from yorishiro.audio.vad import VadRunner
from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.project import Project
from yorishiro.tasks.film.audio import (
    FilmAudioDiarizeStep,
    FilmAudioEmotionStep,
    FilmAudioSTTStep,
    FilmAudioVADStep,
    FilmAudioDiarizeTask,
    FilmAudioEmotionTask,
    FilmAudioSTTTask,
    FilmAudioVADTask,
)
from yorishiro.tasks.registry import ModelRegistry


class VadRunnerTests(unittest.TestCase):
    def test_run_writes_fallback_segment_when_vad_finds_no_speech(self) -> None:
        fake_module = types.SimpleNamespace(
            load_silero_vad=lambda: object(),
            read_audio=lambda _path: [0.0],
            get_speech_timestamps=lambda *_args, **_kwargs: [],
        )
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(sys.modules, {"silero_vad": fake_module}):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"

            segments = VadRunner().run(audio_path, output_dir)

            self.assertEqual(segments, [{"start": 0.0, "end": float("inf")}])
            saved = json.loads((output_dir / "vad.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["start"], 0.0)


class DiarizerTests(unittest.TestCase):
    def test_run_falls_back_to_single_speaker_when_hf_token_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"

            turns = Diarizer(DiarizerConfig(hf_token_env="MISSING_TOKEN")).run(audio_path, output_dir)

            self.assertEqual(turns, [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}])
            saved = json.loads((output_dir / "diarization.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["speaker"], "SPEAKER_00")

    def test_run_resumes_from_checkpoint_and_saves_speaker_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"HF_TOKEN": "x"}):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8")
            ckpt_dir = output_dir / ".diarization_checkpoints"
            ckpt_dir.mkdir()
            source_mtime = audio_path.stat().st_mtime
            checkpoint = {
                "chunk_idx": 0,
                "source_mtime": source_mtime,
                "turns": [{"speaker": "local-0", "start": 0.0, "end": 1.0}],
                "speakers_local": ["local-0"],
            }
            (ckpt_dir / "chunk_0000.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            np.save(str(ckpt_dir / "chunk_0000.npy"), np.array([[1.0, 2.0]], dtype=np.float32))

            diarizer = Diarizer(DiarizerConfig(hf_token_env="HF_TOKEN"))
            with (
                patch("yorishiro.audio.diarization.sf.info", return_value=SimpleNamespace(duration=1.0, samplerate=16000)),
                patch.object(diarizer, "_load_diarization_pipeline", return_value=object()),
                patch.object(diarizer, "_diarize_chunk", side_effect=AssertionError("checkpoint should be used")),
                patch("yorishiro.audio.diarization.save_speaker_bank") as save_bank,
            ):
                turns = diarizer.run(audio_path, output_dir)

            self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
            self.assertTrue(save_bank.called)


class TranscriberTests(unittest.TestCase):
    def test_run_uses_checkpointed_chunk_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            sys.modules,
            {"librosa": types.SimpleNamespace(resample=lambda audio, **_kwargs: audio)},
        ):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8")
            (output_dir / "diarization.json").write_text(
                json.dumps([{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]),
                encoding="utf-8",
            )
            ckpt_dir = output_dir / ".stt_checkpoints"
            ckpt_dir.mkdir()
            source_mtime = audio_path.stat().st_mtime
            checkpoint = {
                "chunk_idx": 0,
                "source_mtime": source_mtime,
                "entries": [
                    {
                        "speaker_global": "SPEAKER_00",
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "confidence": 0.9,
                    }
                ],
                "detected_language": "en",
            }
            (ckpt_dir / "chunk_0000.json").write_text(json.dumps(checkpoint), encoding="utf-8")

            class FakeSoundFile:
                samplerate = 16000
                frames = 16000

                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def __enter__(self) -> FakeSoundFile:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                def seek(self, _offset: int) -> None:
                    return None

                def read(self, _frames: int, dtype: str = "float32") -> np.ndarray:
                    del dtype
                    return np.zeros(16000, dtype=np.float32)

            class ImmediateFuture:
                def __init__(self, result: object) -> None:
                    self._result = result

                def result(self) -> object:
                    return self._result

            class ImmediateExecutor:
                def __init__(self, max_workers: int) -> None:
                    self.max_workers = max_workers

                def __enter__(self) -> ImmediateExecutor:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                def submit(self, fn, *args):  # type: ignore[no-untyped-def]
                    return ImmediateFuture(fn(*args))

            transcriber = Transcriber(TranscriberConfig())
            fake_whisper = object()
            with (
                patch("yorishiro.audio.transcription.sf.SoundFile", FakeSoundFile),
                patch("yorishiro.audio.transcription.get_whisper_model", return_value=fake_whisper),
                patch("yorishiro.audio.transcription.ThreadPoolExecutor", ImmediateExecutor),
            ):
                transcript = transcriber.run(audio_path, output_dir)

            self.assertEqual(transcript.language, "en")
            self.assertEqual(transcript.entries[0].text, "hello")
            saved = json.loads((output_dir / "transcript_raw.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["entries"][0]["text"], "hello")

    def test_split_entry_text_uses_overlap_assignment(self) -> None:
        transcriber = Transcriber()
        entries = transcriber._split_entry_text(
            0.0,
            2.0,
            "Alpha。Beta。",
            0.9,
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
            ],
            "ja",
        )

        self.assertEqual([entry.text for entry in entries], ["Alpha。", "Beta。"])
        self.assertEqual([entry.speaker_global for entry in entries], ["SPEAKER_00", "SPEAKER_01"])


class EmotionAnalyzerTests(unittest.TestCase):
    def test_run_writes_transcript_json_after_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            transcript = Transcript(
                language="ja",
                entries=[
                    TranscriptEntry(
                        speaker_global="SPEAKER_00",
                        start=0.0,
                        end=1.0,
                        text="hello",
                        confidence=0.9,
                    )
                ],
            )
            (output_dir / "transcript_raw.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

            analyzer = EmotionAnalyzer()

            def fake_analyze_emotions(_audio_path: Path, transcript_in: Transcript) -> Transcript:
                transcript_in.entries[0].emotion = "happy"
                return transcript_in

            def fake_analyze_prosody(_audio_path: Path, transcript_in: Transcript) -> Transcript:
                transcript_in.entries[0].volume = "normal"
                return transcript_in

            with (
                patch.object(analyzer, "_analyze_emotions", side_effect=fake_analyze_emotions),
                patch.object(analyzer, "_analyze_prosody", side_effect=fake_analyze_prosody),
            ):
                result = analyzer.run(audio_path, output_dir)

            self.assertEqual(result.entries[0].emotion, "happy")
            saved = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["entries"][0]["emotion"], "happy")
            self.assertEqual(saved["entries"][0]["volume"], "normal")


class FilmAudioStepTests(unittest.TestCase):
    def test_audio_steps_build_expected_tasks_and_preserve_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "raw").mkdir()
            (root / "project.yaml").write_text(
                """project:
  name: Demo
  code: demo
sources:
  - id: film-src
    type: film
    path: raw/demo.mp4
    config:
      language: ja
steps:
  film.audio.vad: {}
  film.audio.diarize: {}
  film.audio.stt: {}
  film.audio.emotion: {}
""",
                encoding="utf-8",
            )
            project = Project.load(root)
            registry = ModelRegistry(project)

            vad_task = FilmAudioVADStep(project, "film-src", registry).tasks()[0]
            diarize_task = FilmAudioDiarizeStep(project, "film-src", registry).tasks()[0]
            stt_task = FilmAudioSTTStep(project, "film-src", registry).tasks()[0]
            emotion_task = FilmAudioEmotionStep(project, "film-src", registry).tasks()[0]

            self.assertIsInstance(vad_task, FilmAudioVADTask)
            self.assertIsInstance(diarize_task, FilmAudioDiarizeTask)
            self.assertIsInstance(stt_task, FilmAudioSTTTask)
            self.assertIsInstance(emotion_task, FilmAudioEmotionTask)
            assert isinstance(stt_task, FilmAudioSTTTask)
            self.assertEqual(stt_task._language, "ja")
            self.assertEqual(vad_task.output_paths(), [project.step_dir("film-src", "audio") / "vad.json"])
            self.assertEqual(
                emotion_task.output_paths(),
                [project.step_dir("film-src", "audio") / "transcript.json"],
            )
