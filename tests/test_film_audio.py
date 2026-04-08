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
from yorishiro.audio.speaker_attribution import SpeakerAttributor, SpeakerAttributorConfig
from yorishiro.audio.transcription import SpeechGroup, Transcriber, TranscriberConfig
from yorishiro.audio.vad import VadRunner
from yorishiro.models.film_models import STTEntry, SpeakerAttribution, SpeakerAttributionEntry, STTTranscript, Transcript
from yorishiro.project import Project
from yorishiro.tasks.film.audio import (
    FilmAudioEmotionStep,
    FilmAudioSpeakersStep,
    FilmAudioSTTStep,
    FilmAudioVADStep,
    FilmAudioEmotionTask,
    FilmAudioSpeakersTask,
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
            ckpt_dir = output_dir / ".stt_checkpoints"
            ckpt_dir.mkdir()
            source_mtime = audio_path.stat().st_mtime
            checkpoint = {
                "groups": [
                    {
                        "group_id": "g_000000_000000",
                        "span_start_idx": 0,
                        "span_end_idx": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "source_mtime": source_mtime,
                        "entries": [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "text": "hello",
                                "confidence": 0.9,
                            }
                        ],
                        "detected_language": "en",
                    }
                ]
            }
            (ckpt_dir / "groups_0000.json").write_text(json.dumps(checkpoint), encoding="utf-8")

            class FakeSoundFile:
                samplerate = 16000
                frames = 16000

                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def __enter__(self) -> FakeSoundFile:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            transcriber = Transcriber(TranscriberConfig())
            with (
                patch("yorishiro.audio.transcription.sf.SoundFile", FakeSoundFile),
                patch("yorishiro.audio.transcription.torch.cuda.is_available", return_value=False),
            ):
                transcript = transcriber.run(audio_path, output_dir)

            self.assertEqual(transcript.language, "en")
            self.assertEqual(transcript.entries[0].text, "hello")
            saved = json.loads((output_dir / "stt.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["entries"][0]["text"], "hello")

    def test_run_uses_vad_segment_offsets_for_transcript_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            sys.modules,
            {"librosa": types.SimpleNamespace(resample=lambda audio, **_kwargs: audio)},
        ):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(json.dumps([{"start": 26.1, "end": 27.5}]), encoding="utf-8")

            class FakeSoundFile:
                samplerate = 16000
                frames = 30 * 16000

                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def __enter__(self) -> FakeSoundFile:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            class FakeSegment:
                def __init__(self) -> None:
                    self.start = 0.0
                    self.end = 1.4
                    self.text = "hello"
                    self.avg_logprob = -0.1
                    self.words = []

            class FakeWhisper:
                def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
                    del audio, kwargs
                    return iter([FakeSegment()]), SimpleNamespace(language="en")

            transcriber = Transcriber(TranscriberConfig())
            with (
                patch("yorishiro.audio.transcription.sf.SoundFile", FakeSoundFile),
                patch(
                    "yorishiro.audio.transcription.sf.read",
                    return_value=(np.zeros(int(1.4 * 16000), dtype=np.float32), 16000),
                ),
                patch("yorishiro.audio.transcription.get_whisper_model", return_value=FakeWhisper()),
                patch("yorishiro.audio.transcription.torch.cuda.is_available", return_value=False),
            ):
                transcript = transcriber.run(audio_path, output_dir)

            self.assertEqual(len(transcript.entries), 1)
            self.assertEqual(transcript.entries[0].start, 26.1)
            self.assertEqual(transcript.entries[0].end, 27.5)
            self.assertEqual(transcript.entries[0].text, "hello")

    def test_run_worker_groups_saves_each_completed_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")

            class FakeSegment:
                def __init__(self, end: float, text: str) -> None:
                    self.start = 0.0
                    self.end = end
                    self.text = text
                    self.avg_logprob = -0.1
                    self.words = []

            class FakeWhisper:
                def __init__(self) -> None:
                    self.calls = 0

                def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
                    del audio, kwargs
                    self.calls += 1
                    if self.calls == 1:
                        return iter([FakeSegment(1.0, "one")]), SimpleNamespace(language="en")
                    return iter([FakeSegment(0.5, "two")]), SimpleNamespace(language="en")

            saved_group_ids: list[str] = []
            transcriber = Transcriber(TranscriberConfig())
            groups = [
                SpeechGroup(group_id="g_000000_000000", span_start_idx=0, span_end_idx=0, start=0.0, end=1.0),
                SpeechGroup(group_id="g_000001_000001", span_start_idx=1, span_end_idx=1, start=2.0, end=2.5),
            ]
            fake_whisper = FakeWhisper()
            with (
                patch(
                    "yorishiro.audio.transcription.sf.read",
                    side_effect=[
                        (np.zeros(16000, dtype=np.float32), 16000),
                        (np.zeros(8000, dtype=np.float32), 16000),
                    ],
                ),
                patch("yorishiro.audio.transcription.get_whisper_model", return_value=fake_whisper),
            ):
                results = transcriber._run_worker_groups(
                    groups,
                    worker_idx=0,
                    worker_config=TranscriberConfig(),
                    audio_path=audio_path,
                    file_sample_rate=16000,
                    language="en",
                    librosa_module=types.SimpleNamespace(resample=lambda audio, **_kwargs: audio),
                    update_progress=lambda _delta: None,
                    save_result=lambda result: saved_group_ids.append(result.group_id),
                )

            self.assertEqual([result.group_id for result in results], ["g_000000_000000", "g_000001_000001"])
            self.assertEqual(saved_group_ids, ["g_000000_000000", "g_000001_000001"])

    def test_split_entry_text_preserves_pieces(self) -> None:
        transcriber = Transcriber()
        entries = transcriber._split_entry_text(
            0.0,
            2.0,
            "Alpha。Beta。",
            0.9,
            "ja",
        )

        self.assertEqual([entry.text for entry in entries], ["Alpha。", "Beta。"])
        self.assertEqual(entries[0].start, 0.0)
        self.assertGreater(entries[1].start, entries[0].start)
        self.assertEqual(entries[-1].end, 2.0)

    def test_segment_to_entries_filters_low_confidence(self) -> None:
        transcriber = Transcriber(TranscriberConfig(stt_min_confidence=-0.5))
        segment = SimpleNamespace(
            text="hello",
            start=0.0,
            end=1.0,
            avg_logprob=-0.9,
        )

        entries = transcriber._segment_to_entries(
            segment,
            0.0,
            "en",
        )

        self.assertEqual(entries, [])

    def test_segment_to_entries_filters_impossible_chars_per_second(self) -> None:
        transcriber = Transcriber(TranscriberConfig(stt_max_chars_per_second=20.0))
        segment = SimpleNamespace(
            text="これは不自然に長いテキストです",
            start=0.0,
            end=0.1,
            avg_logprob=-0.2,
        )

        entries = transcriber._segment_to_entries(
            segment,
            0.0,
            "ja",
        )

        self.assertEqual(entries, [])


class SpeakerAttributorTests(unittest.TestCase):
    def test_run_writes_attribution_and_speaker_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            stt = STTTranscript(
                language="ja",
                entries=[
                    STTEntry(start=0.0, end=1.2, text="a", confidence=0.1),
                    STTEntry(start=1.5, end=2.8, text="b", confidence=0.2),
                ],
            )
            (output_dir / "stt.json").write_text(stt.model_dump_json(indent=2), encoding="utf-8")

            emb1 = np.array([1.0, 0.0], dtype=np.float32)
            emb2 = np.array([0.99, 0.01], dtype=np.float32)
            attributor = SpeakerAttributor(SpeakerAttributorConfig())
            with patch(
                "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                side_effect=[emb1, emb2],
            ):
                result = attributor.run(audio_path, output_dir)

            self.assertEqual([entry.speaker_id for entry in result.entries], ["SPKR_001", "SPKR_001"])
            saved = SpeakerAttribution(**json.loads((output_dir / "speaker_attribution.json").read_text(encoding="utf-8")))
            self.assertEqual(saved.entries[0].speaker_id, "SPKR_001")
            self.assertTrue((output_dir / "speaker_bank.json").exists())


class EmotionAnalyzerTests(unittest.TestCase):
    def test_run_writes_transcript_json_after_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            stt = STTTranscript(
                language="ja",
                entries=[
                    STTEntry(
                        start=0.0,
                        end=1.0,
                        text="hello",
                        confidence=0.9,
                    )
                ],
            )
            attribution = SpeakerAttribution(
                entries=[
                    SpeakerAttributionEntry(
                        entry_id="utt_000000",
                        start=0.0,
                        end=1.0,
                        speaker_id="SPKR_001",
                        similarity=0.9,
                        embedding_present=True,
                        enrolled=True,
                    )
                ]
            )
            (output_dir / "stt.json").write_text(stt.model_dump_json(indent=2), encoding="utf-8")
            (output_dir / "speaker_attribution.json").write_text(attribution.model_dump_json(indent=2), encoding="utf-8")

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
  film.audio.stt: {}
  film.audio.speakers: {}
  film.audio.emotion: {}
""",
                encoding="utf-8",
            )
            project = Project.load(root)
            registry = ModelRegistry(project)

            vad_task = FilmAudioVADStep(project, "film-src", registry).tasks()[0]
            stt_task = FilmAudioSTTStep(project, "film-src", registry).tasks()[0]
            speakers_task = FilmAudioSpeakersStep(project, "film-src", registry).tasks()[0]
            emotion_task = FilmAudioEmotionStep(project, "film-src", registry).tasks()[0]

            self.assertIsInstance(vad_task, FilmAudioVADTask)
            self.assertIsInstance(stt_task, FilmAudioSTTTask)
            self.assertIsInstance(speakers_task, FilmAudioSpeakersTask)
            self.assertIsInstance(emotion_task, FilmAudioEmotionTask)
            assert isinstance(stt_task, FilmAudioSTTTask)
            self.assertEqual(stt_task._language, "ja")
            self.assertEqual(vad_task.output_paths(), [project.step_dir("film-src", "audio") / "vad.json"])
            self.assertEqual(speakers_task.output_paths(), [project.step_dir("film-src", "audio") / "speaker_attribution.json"])
            self.assertEqual(
                emotion_task.output_paths(),
                [project.step_dir("film-src", "audio") / "transcript.json"],
            )
