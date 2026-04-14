from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import torch

from yorishiro.audio.diarization import Diarizer, DiarizerConfig
from yorishiro.audio.emotion_analysis import EmotionAnalyzer
from yorishiro.audio.separator import AudioSeparator
from yorishiro.audio.speaker_bank import SpeakerBankManager
from yorishiro.audio.speaker_attribution import (
    ClusteringScoreBreakdown,
    EmbeddingWindow,
    SpeakerAttributor,
    SpeakerAttributorConfig,
    SweepCandidateRow,
)
from yorishiro.audio.transcription import (
    _AlignedToken,
    GroupResult,
    SpeechSpan,
    SpeechGroup,
    split_with_aligned_tokens,
    Transcriber,
    TranscriberConfig,
)
from yorishiro.audio._speech_support import (
    compute_avg_logprob_from_captured_logits,
    FunASRConfidenceHook,
    Qwen3AlignerConfidenceHook,
)
from yorishiro.audio.vad import VadConfig, VadRunner
from yorishiro.models.film_models import (
    STTEntry,
    SpeakerAttribution,
    SpeakerAttributionEntry,
    STTTranscript,
    Transcript,
)
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
from yorishiro.tasks.registry import ModelRegistry, StepRuntime


class VadRunnerTests(unittest.TestCase):
    def test_run_writes_vad_and_debug_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            expected_segments = [{"start": 0.123, "end": 0.789, "quality": 0.91}]
            expected_debug = {
                "audio_duration": 1.23,
                "profile": "balanced",
                "primary_segments": [],
                "rescue_segments_added": [],
                "dropped_segments": [],
                "adjustments": [],
                "stats": {
                    "primary_count": 0,
                    "rescue_added_count": 0,
                    "dropped_count": 0,
                    "final_count": 1,
                    "total_speech_seconds": 0.666,
                    "mean_quality": 0.91,
                    "low_quality_count": 0,
                    "largest_boundary_shift_ms": 0.0,
                },
            }
            runner = VadRunner()
            with patch.object(
                runner, "_run_vad", return_value=(expected_segments, expected_debug)
            ):
                segments = runner.run(audio_path, output_dir)

            self.assertEqual(segments, expected_segments)
            saved_vad = json.loads(
                (output_dir / "vad.json").read_text(encoding="utf-8")
            )
            saved_debug = json.loads(
                (output_dir / "vad.debug.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_vad, expected_segments)
            self.assertEqual(saved_debug, expected_debug)

    def test_post_process_repairs_overlap_without_merging(self) -> None:
        runner = VadRunner()
        segments, dropped = runner._post_process_segments(
            [
                {"start": 1.0, "end": 1.5, "quality": 0.8},
                {"start": 1.4, "end": 1.9, "quality": 0.7},
            ],
            audio_duration=3.0,
        )

        self.assertEqual(len(dropped), 0)
        self.assertEqual(len(segments), 2)
        self.assertLessEqual(segments[0]["end"], segments[1]["start"])

    def test_add_rescue_segments_only_keeps_isolated_candidates(self) -> None:
        runner = VadRunner(VadConfig(vad_profile="balanced"))
        merged, added = runner._add_rescue_segments(
            base_segments=[{"start": 1.0, "end": 1.5, "quality": 0.8}],
            rescue_segments=[
                {"start": 1.55, "end": 1.8, "quality": 0.6, "source": "mixed-rescue"},
                {"start": 3.0, "end": 3.3, "quality": 0.5, "source": "mixed-rescue"},
            ],
            guard_gap_seconds=0.2,
            audio_duration=5.0,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(len(added), 1)
        self.assertAlmostEqual(float(added[0]["start"]), 3.0)


class DiarizerTests(unittest.TestCase):
    def test_run_falls_back_to_single_speaker_when_hf_token_missing(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.dict(os.environ, {}, clear=True),
        ):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"

            turns = Diarizer(DiarizerConfig(hf_token_env="MISSING_TOKEN")).run(
                audio_path, output_dir
            )

            self.assertEqual(
                turns, [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]
            )
            saved = json.loads(
                (output_dir / "diarization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved[0]["speaker"], "SPEAKER_00")

    def test_run_resumes_from_checkpoint_and_saves_speaker_bank(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.dict(os.environ, {"HF_TOKEN": "x"}),
        ):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(
                json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
            )
            ckpt_dir = output_dir / ".diarization_checkpoints"
            ckpt_dir.mkdir()
            source_mtime = audio_path.stat().st_mtime
            checkpoint = {
                "chunk_idx": 0,
                "source_mtime": source_mtime,
                "turns": [{"speaker": "local-0", "start": 0.0, "end": 1.0}],
                "speakers_local": ["local-0"],
            }
            (ckpt_dir / "chunk_0000.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )
            np.save(
                str(ckpt_dir / "chunk_0000.npy"),
                np.array([[1.0, 2.0]], dtype=np.float32),
            )

            diarizer = Diarizer(DiarizerConfig(hf_token_env="HF_TOKEN"))
            with (
                patch(
                    "yorishiro.audio.diarization.sf.info",
                    return_value=SimpleNamespace(duration=1.0, samplerate=16000),
                ),
                patch.object(
                    diarizer, "_load_diarization_pipeline", return_value=object()
                ),
                patch.object(
                    diarizer,
                    "_diarize_chunk",
                    side_effect=AssertionError("checkpoint should be used"),
                ),
                patch("yorishiro.audio.diarization.save_speaker_bank") as save_bank,
            ):
                turns = diarizer.run(audio_path, output_dir)

            self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
            self.assertTrue(save_bank.called)


class TranscriberTests(unittest.TestCase):
    def test_assemble_transcript_assigns_entry_ids(self) -> None:
        transcriber = Transcriber()
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=1.0,
            )
        ]
        result = transcriber._assemble_transcript(
            groups,
            {
                "g_000000_000000": GroupResult(
                    group_id="g_000000_000000",
                    span_start_idx=0,
                    span_end_idx=0,
                    start=0.0,
                    end=1.0,
                    entries=[
                        {
                            "entry_id": "",
                            "start": 0.0,
                            "end": 1.0,
                            "text": "a",
                            "confidence": 0.9,
                        }
                    ],
                    detected_language="ja",
                    source_mtime=0.0,
                )
            },
            language=None,
        )
        self.assertEqual(result.entries[0].entry_id, "utt_000000")

    def test_run_resumes_matching_checkpoints_and_cleans_them_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(
                json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
            )
            ckpt_dir = output_dir / ".stt_checkpoints"
            ckpt_dir.mkdir()
            transcriber = Transcriber(TranscriberConfig())
            input_signature = transcriber._input_signature(
                audio_path,
                output_dir / "vad.json",
                None,
            )
            (ckpt_dir / "meta.json").write_text(
                json.dumps({"input_signature": input_signature}),
                encoding="utf-8",
            )
            checkpoint = {
                "groups": [
                    {
                        "group_id": "g_000000_000000",
                        "span_start_idx": 0,
                        "span_end_idx": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "source_mtime": audio_path.stat().st_mtime,
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
            (ckpt_dir / "groups_0000.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )

            class FakeSoundFile:
                samplerate = 16000
                frames = 16000

                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def __enter__(self) -> FakeSoundFile:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            with (
                patch("yorishiro.audio.transcription.sf.SoundFile", FakeSoundFile),
                patch(
                    "yorishiro.audio.transcription.get_whisper_model",
                    side_effect=AssertionError("checkpoint should be used"),
                ),
                patch(
                    "yorishiro.audio.transcription.torch.cuda.is_available",
                    return_value=False,
                ),
            ):
                transcript = transcriber.run(audio_path, output_dir)

            self.assertEqual(transcript.language, "en")
            self.assertEqual(transcript.entries[0].text, "hello")
            saved = json.loads((output_dir / "stt.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["entries"][0]["text"], "hello")
            self.assertFalse(ckpt_dir.exists())

    def test_force_run_removes_existing_stt_output_before_transcribing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(
                json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
            )
            stt_path = output_dir / "stt.json"
            stt_path.write_text(
                json.dumps({"language": "ja", "entries": [{"text": "old"}]}),
                encoding="utf-8",
            )

            transcriber = Transcriber(TranscriberConfig())
            with patch.object(
                transcriber,
                "_run_transcription",
                side_effect=KeyboardInterrupt(),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    transcriber.run(audio_path, output_dir, force=True)

            self.assertFalse(stt_path.exists())

    def test_run_discards_stale_checkpoints_and_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            vad_path = output_dir / "vad.json"
            vad_path.write_text(
                json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
            )
            ckpt_dir = output_dir / ".stt_checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "meta.json").write_text(
                json.dumps({"input_signature": "stale-signature"}),
                encoding="utf-8",
            )
            (ckpt_dir / "groups_0000.json").write_text(
                json.dumps({"groups": []}),
                encoding="utf-8",
            )

            class FakeSoundFile:
                samplerate = 16000
                frames = 16000

                def __init__(self, *_args, **_kwargs) -> None:
                    pass

                def __enter__(self) -> FakeSoundFile:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            class FakeSegment:
                def __init__(self) -> None:
                    self.start = 0.0
                    self.end = 1.0
                    self.text = "fresh"
                    self.avg_logprob = -0.1
                    self.words = []

            class FakeWhisper:
                def __init__(self) -> None:
                    self.calls = 0

                def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
                    del audio, kwargs
                    self.calls += 1
                    return iter([FakeSegment()]), SimpleNamespace(language="en")

            transcriber = Transcriber(TranscriberConfig())
            fake_whisper = FakeWhisper()
            with (
                patch("yorishiro.audio.transcription.sf.SoundFile", FakeSoundFile),
                patch(
                    "yorishiro.audio.transcription.sf.read",
                    return_value=(np.zeros(16000, dtype=np.float32), 16000),
                ),
                patch(
                    "yorishiro.audio.transcription.get_whisper_model",
                    return_value=fake_whisper,
                ),
                patch(
                    "yorishiro.audio.transcription.torch.cuda.is_available",
                    return_value=False,
                ),
            ):
                transcript = transcriber.run(audio_path, output_dir)

            self.assertEqual(transcript.entries[0].text, "fresh")
            self.assertEqual(fake_whisper.calls, 1)
            self.assertFalse(ckpt_dir.exists())

    def test_run_refuses_to_publish_stale_output_when_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(
                json.dumps([{"start": 0.0, "end": 1.0}]), encoding="utf-8"
            )

            transcriber = Transcriber(TranscriberConfig())
            with (
                patch.object(
                    transcriber,
                    "_run_transcription",
                    return_value=STTTranscript(language="en", entries=[]),
                ),
                patch.object(
                    transcriber,
                    "_input_signature",
                    side_effect=["sig-a", "sig-b"],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                    transcriber.run(audio_path, output_dir)

            self.assertFalse((output_dir / "stt.json").exists())

    def test_run_uses_vad_segment_offsets_for_transcript_timestamps(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.dict(
                sys.modules,
                {
                    "librosa": types.SimpleNamespace(
                        resample=lambda audio, **_kwargs: audio
                    )
                },
            ),
        ):
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            (output_dir / "vad.json").write_text(
                json.dumps([{"start": 26.1, "end": 27.5}]), encoding="utf-8"
            )

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
                patch(
                    "yorishiro.audio.transcription.get_whisper_model",
                    return_value=FakeWhisper(),
                ),
                patch(
                    "yorishiro.audio.transcription.torch.cuda.is_available",
                    return_value=False,
                ),
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
                        return iter([FakeSegment(1.0, "one")]), SimpleNamespace(
                            language="en"
                        )
                    return iter([FakeSegment(0.5, "two")]), SimpleNamespace(
                        language="en"
                    )

            saved_group_ids: list[str] = []
            transcriber = Transcriber(TranscriberConfig())
            groups = [
                SpeechGroup(
                    group_id="g_000000_000000",
                    span_start_idx=0,
                    span_end_idx=0,
                    start=0.0,
                    end=1.0,
                ),
                SpeechGroup(
                    group_id="g_000001_000001",
                    span_start_idx=1,
                    span_end_idx=1,
                    start=2.0,
                    end=2.5,
                ),
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
                patch(
                    "yorishiro.audio.transcription.get_whisper_model",
                    return_value=fake_whisper,
                ),
            ):
                results = transcriber._run_worker_groups(
                    groups,
                    worker_idx=0,
                    worker_config=TranscriberConfig(),
                    audio_path=audio_path,
                    file_sample_rate=16000,
                    language="en",
                    resample_module=types.SimpleNamespace(
                        resample=lambda audio, **_kwargs: audio
                    ),
                    update_progress=lambda _delta: None,
                    save_result=lambda result: saved_group_ids.append(result.group_id),
                )

            self.assertEqual(
                [result.group_id for result in results],
                ["g_000000_000000", "g_000001_000001"],
            )
            self.assertEqual(saved_group_ids, ["g_000000_000000", "g_000001_000001"])

    def test_save_group_results_batch_persists_multiple_groups(self) -> None:
        transcriber = Transcriber(TranscriberConfig(stt_checkpoint_shard_size=2))
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / ".stt_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            transcriber._save_group_results_batch(
                checkpoint_dir,
                [
                    GroupResult(
                        group_id="g_000000_000000",
                        span_start_idx=0,
                        span_end_idx=0,
                        start=0.0,
                        end=1.0,
                        entries=[],
                        detected_language="en",
                        source_mtime=1.0,
                    ),
                    GroupResult(
                        group_id="g_000001_000001",
                        span_start_idx=1,
                        span_end_idx=1,
                        start=1.0,
                        end=2.0,
                        entries=[],
                        detected_language="en",
                        source_mtime=1.0,
                    ),
                ],
            )
            payload = json.loads(
                (checkpoint_dir / "groups_0000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["group_id"] for item in payload["groups"]],
                ["g_000000_000000", "g_000001_000001"],
            )

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

    def test_split_with_aligned_tokens_splits_on_hard_pause(self) -> None:
        segments = split_with_aligned_tokens(
            "Hello there",
            [
                _AlignedToken(text="Hello", start=0.0, end=1.6, confidence=0.9),
                _AlignedToken(text="there", start=2.5, end=3.8, confidence=0.9),
            ],
            0.0,
            3.8,
            "en",
        )

        self.assertEqual(segments, [("Hello", 0.0, 1.6), ("there", 2.5, 3.8)])

    def test_split_with_aligned_tokens_soft_pause_merges_short_duration(self) -> None:
        segments = split_with_aligned_tokens(
            "色派のエイムすっげぇ。",
            [
                _AlignedToken(text="色派", start=79.464, end=79.464, confidence=0.9),
                _AlignedToken(text="の", start=79.784, end=79.944, confidence=0.9),
                _AlignedToken(text="エイム", start=79.944, end=80.264, confidence=0.9),
                _AlignedToken(
                    text="すっげぇ", start=80.824, end=80.904, confidence=0.9
                ),
            ],
            79.144,
            84.696,
            "ja",
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], "色派のエイムすっげぇ。")

    def test_split_with_aligned_tokens_soft_pause_splits_when_both_sides_long(
        self,
    ) -> None:
        segments = split_with_aligned_tokens(
            "Hello there friend",
            [
                _AlignedToken(text="Hello", start=0.0, end=2.0, confidence=0.9),
                _AlignedToken(text="there", start=2.0, end=4.0, confidence=0.9),
                _AlignedToken(text="friend", start=4.4, end=6.0, confidence=0.9),
            ],
            0.0,
            6.0,
            "en",
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0], "Hello there")
        self.assertEqual(segments[1][0], "friend")

    def test_split_with_aligned_tokens_keeps_short_soft_pause_together(self) -> None:
        segments = split_with_aligned_tokens(
            "Ahh well",
            [
                _AlignedToken(text="Ahh", start=0.0, end=0.2, confidence=0.9),
                _AlignedToken(text="well", start=0.6, end=0.9, confidence=0.9),
            ],
            0.0,
            0.9,
            "en",
        )

        self.assertEqual(segments, [("Ahh well", 0.0, 0.9)])

    def test_split_with_aligned_tokens_respects_sentence_boundary_without_pause(
        self,
    ) -> None:
        segments = split_with_aligned_tokens(
            "Mr. Ben had this issue. She replied.",
            [
                _AlignedToken(text="Mr.", start=0.0, end=0.1, confidence=0.9),
                _AlignedToken(text="Ben", start=0.1, end=0.3, confidence=0.9),
                _AlignedToken(text="had", start=0.3, end=0.45, confidence=0.9),
                _AlignedToken(text="this", start=0.45, end=0.6, confidence=0.9),
                _AlignedToken(text="issue.", start=0.6, end=0.9, confidence=0.9),
                _AlignedToken(text="She", start=0.92, end=1.05, confidence=0.9),
                _AlignedToken(text="replied.", start=1.05, end=1.35, confidence=0.9),
            ],
            0.0,
            1.35,
            "en",
        )

        self.assertEqual(
            segments,
            [
                ("Mr. Ben had this issue.", 0.0, 0.9),
                ("She replied.", 0.92, 1.35),
            ],
        )

    def test_split_with_aligned_tokens_maps_punctuation_stripped_alignment_space(
        self,
    ) -> None:
        segments = split_with_aligned_tokens(
            "Hello, world! How are you?",
            [
                _AlignedToken(text="Hello", start=0.0, end=0.2, confidence=0.9),
                _AlignedToken(text="world", start=0.2, end=0.5, confidence=0.9),
                _AlignedToken(text="How", start=0.55, end=0.7, confidence=0.9),
                _AlignedToken(text="are", start=0.7, end=0.85, confidence=0.9),
                _AlignedToken(text="you", start=0.85, end=1.0, confidence=0.9),
            ],
            0.0,
            1.0,
            "en",
        )

        self.assertEqual(
            segments,
            [
                ("Hello, world!", 0.0, 0.5),
                ("How are you?", 0.55, 1.0),
            ],
        )

    def test_split_with_aligned_tokens_does_not_create_punctuation_only_segment(
        self,
    ) -> None:
        segments = split_with_aligned_tokens(
            "これかぐや姫絶対不幸じゃん! 次どうする?",
            [
                _AlignedToken(
                    text="これかぐや姫絶対不幸じゃん",
                    start=0.0,
                    end=1.6,
                    confidence=0.9,
                ),
                _AlignedToken(text="次どうする", start=2.4, end=3.0, confidence=0.9),
            ],
            0.0,
            3.2,
            "ja",
        )

        self.assertEqual(
            segments,
            [
                ("これかぐや姫絶対不幸じゃん!", 0.0, 1.6),
                ("次どうする?", 2.4, 3.0),
            ],
        )

    def test_segment_to_entries_prefers_word_timestamps_for_pause_split(self) -> None:
        transcriber = Transcriber()
        segment = SimpleNamespace(
            text="Hello there",
            start=0.0,
            end=1.2,
            avg_logprob=-0.1,
            words=[
                SimpleNamespace(start=0.0, end=0.2, word="Hello"),
                SimpleNamespace(start=0.9, end=1.2, word="there"),
            ],
        )

        entries = transcriber._segment_to_entries(segment, 0.0, 1.2, "en")

        self.assertEqual([entry.text for entry in entries], ["Hello there"])
        self.assertEqual((entries[0].start, entries[0].end), (0.0, 1.2))

    def test_segment_to_entries_filters_low_confidence(self) -> None:
        transcriber = Transcriber()
        segment = SimpleNamespace(
            text="hello",
            start=0.0,
            end=1.0,
            avg_logprob=-0.9,
        )

        entries = transcriber._segment_to_entries(
            segment,
            0.0,
            1.0,
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
            0.1,
            "ja",
        )

        self.assertEqual(entries, [])

    def test_speech_spans_filter_too_short(self) -> None:
        transcriber = Transcriber(TranscriberConfig(stt_min_segment_seconds=0.2))
        spans = transcriber._speech_spans(
            [
                {"start": 1.0, "end": 1.1},
                {"start": 2.0, "end": 2.3},
            ],
            total_duration=10.0,
        )
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0].start, 2.0)
        self.assertAlmostEqual(spans[0].end, 2.3)

    def test_segment_to_entries_clamps_to_chunk_and_filters_too_short(self) -> None:
        transcriber = Transcriber(TranscriberConfig(stt_min_segment_seconds=0.2))
        segment = SimpleNamespace(
            text="ごめん",
            start=0.0,
            end=2.0,
            avg_logprob=-0.1,
        )
        # chunk is only 0.15s wide after clamping -> dropped by min_segment_seconds
        entries = transcriber._segment_to_entries(
            segment,
            41.9,
            42.05,
            "ja",
        )
        self.assertEqual(entries, [])

    def test_transformers_chunks_to_segments_converts_timestamps(self) -> None:
        chunks = [
            {"text": " hello", "timestamp": (0.0, 1.5)},
            {"text": " world", "timestamp": (1.5, 2.8)},
        ]
        segments = Transcriber._transformers_chunks_to_segments(
            chunks, 10.0, 15.0, "en"
        )
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 1.5)
        self.assertEqual(segments[0].text, "hello")
        self.assertEqual(segments[1].text, "world")

    def test_transformers_chunks_to_segments_skips_empty(self) -> None:
        chunks = [
            {"text": "hello", "timestamp": (0.0, 1.0)},
            {"text": "", "timestamp": (1.0, 2.0)},
            {"text": "  ", "timestamp": (2.0, 3.0)},
        ]
        segments = Transcriber._transformers_chunks_to_segments(chunks, 0.0, 5.0, "en")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "hello")

    def test_transformers_chunks_to_segments_handles_missing_timestamps(self) -> None:
        chunks = [
            {"text": "no timestamps"},
        ]
        segments = Transcriber._transformers_chunks_to_segments(chunks, 0.0, 5.0, "en")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 0.0)

    def test_transformers_backend_dispatches_to_transformers_path(self) -> None:
        config = TranscriberConfig(
            stt_backend="transformers-whisper",
            stt_model="kotoba-tech/kotoba-whisper-v2.1",
            stt_extra_args={"batch_size": 8},
        )
        transcriber = Transcriber(config)
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=1.0,
            ),
        ]

        class FakePipeline:
            def __call__(self, audio, **kwargs):
                del audio, kwargs
                return {
                    "text": "hello world",
                    "chunks": [
                        {"text": " hello", "timestamp": (0.0, 0.5)},
                        {"text": " world", "timestamp": (0.5, 1.0)},
                    ],
                }

        fake_pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "test.wav"
            audio_path.write_bytes(b"stub")
            with (
                patch(
                    "yorishiro.audio.transcription.get_transformers_pipeline",
                    return_value=fake_pipeline,
                ),
                patch(
                    "yorishiro.audio.transcription.sf.read",
                    return_value=(np.zeros(16000, dtype=np.float32), 16000),
                ),
                patch("yorishiro.audio.transcription.sf.SoundFile") as FakeSoundFile,
            ):
                FakeSoundFile.return_value.__enter__ = lambda s: SimpleNamespace(
                    samplerate=16000, frames=16000
                )
                FakeSoundFile.return_value.__exit__ = lambda s, *a: None
                results = transcriber._run_worker_groups(
                    groups,
                    worker_idx=0,
                    worker_config=config,
                    audio_path=audio_path,
                    file_sample_rate=16000,
                    language="ja",
                    resample_module=types.SimpleNamespace(
                        resample=lambda audio, **_kw: audio
                    ),
                    update_progress=lambda _delta: None,
                    save_result=lambda _r: None,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].detected_language, "ja")

    def test_transformers_chunks_to_segments_uses_group_confidence(self) -> None:
        chunks = [
            {"text": "hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
        segments = Transcriber._transformers_chunks_to_segments(
            chunks,
            0.0,
            1.0,
            "en",
            group_confidence=-0.35,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].avg_logprob, -0.35)
        self.assertEqual(segments[1].avg_logprob, -0.35)

    def test_extract_segment_avg_logprob_from_scores_and_tokens(self) -> None:
        transcriber = Transcriber()
        vocab_size = 4
        score0 = torch.tensor([[-10.0, -10.0, 8.0, -10.0]], dtype=torch.float32)
        score1 = torch.tensor([[-10.0, 9.0, -10.0, -10.0]], dtype=torch.float32)
        segment = {
            "idxs": (2, 4),
            "result": {
                "sequences": torch.tensor([[0, 0, 2, 1]], dtype=torch.long),
                "scores": [score0[:, :vocab_size], score1[:, :vocab_size]],
            },
        }
        confidence = transcriber._extract_segment_avg_logprob(segment)
        self.assertIsNotNone(confidence)
        assert confidence is not None
        self.assertGreater(confidence, -0.001)

    def test_transformers_group_confidence_averages_segment_scores(self) -> None:
        transcriber = Transcriber()
        generated = {
            "segments": [
                [
                    {"result": {"sequences_scores": torch.tensor([-0.2])}},
                    {"result": {"sequences_scores": torch.tensor([-0.6])}},
                ]
            ]
        }
        confidence = transcriber._transformers_group_confidence([generated])
        self.assertIsNotNone(confidence)
        assert confidence is not None
        self.assertAlmostEqual(confidence, -0.4)

    def test_faster_whisper_backend_remains_default(self) -> None:
        config = TranscriberConfig()
        self.assertEqual(config.stt_backend, "faster-whisper")
        self.assertEqual(config.stt_extra_args, {})

    def test_extra_args_default_is_empty_dict(self) -> None:
        config = TranscriberConfig(stt_backend="transformers-whisper")
        self.assertEqual(config.stt_extra_args, {})


class FunASRLanguageMappingTests(unittest.TestCase):
    def test_common_languages_map_to_chinese_names(self) -> None:
        from yorishiro.audio._speech_support import funasr_language

        self.assertEqual(funasr_language("ja"), "日文")
        self.assertEqual(funasr_language("en"), "英文")
        self.assertEqual(funasr_language("zh"), "中文")
        self.assertEqual(funasr_language("ko"), "韩文")

    def test_none_language_returns_auto(self) -> None:
        from yorishiro.audio._speech_support import funasr_language

        self.assertEqual(funasr_language(None), "auto")

    def test_unknown_language_passes_through(self) -> None:
        from yorishiro.audio._speech_support import funasr_language

        self.assertEqual(funasr_language("fr"), "fr")

    def test_normalizes_language_before_mapping(self) -> None:
        from yorishiro.audio._speech_support import funasr_language

        self.assertEqual(funasr_language("ja-JP"), "日文")
        self.assertEqual(funasr_language("en-US"), "英文")


class SmartSplitHeuristicTests(unittest.TestCase):
    def test_keeps_number_commas_intact(self) -> None:
        from yorishiro.audio._speech_support import split_text_heuristically

        self.assertEqual(
            split_text_heuristically("合計で13,243円です。", "ja"),
            ["合計で13,243円です。"],
        )

    def test_handles_english_abbreviations(self) -> None:
        from yorishiro.audio._speech_support import split_text_heuristically

        self.assertEqual(
            split_text_heuristically("Mr. Ben had this issue. She replied.", "en"),
            ["Mr. Ben had this issue.", "She replied."],
        )

    def test_splits_english_clause_boundary_but_not_list_comma(self) -> None:
        from yorishiro.audio._speech_support import split_text_heuristically

        self.assertEqual(
            split_text_heuristically("I think aaaa bbbb, because it is ccc dddd", "en"),
            ["I think aaaa bbbb,", "because it is ccc dddd"],
        )
        self.assertEqual(
            split_text_heuristically("I need to buy egg, milk and noodles", "en"),
            ["I need to buy egg, milk and noodles"],
        )

    def test_splits_cjk_clause_boundaries_without_isolating_short_prefix(self) -> None:
        from yorishiro.audio._speech_support import split_text_heuristically

        self.assertEqual(
            split_text_heuristically("我觉得这个很好，因为它很有用。", "zh"),
            ["我觉得这个很好，", "因为它很有用。"],
        )
        self.assertEqual(
            split_text_heuristically(
                "是的，我觉得不太好，但是我明天会好起来的。", "zh"
            ),
            ["是的，我觉得不太好，", "但是我明天会好起来的。"],
        )


class ConfidenceExtractionTests(unittest.TestCase):
    def test_compute_avg_logprob_returns_none_when_empty(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        captured = _CapturedLogits()
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNone(result)

    def test_compute_avg_logprob_returns_none_when_no_sequences(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        captured = _CapturedLogits(
            sequences_scores=torch.tensor([-3.0]),
        )
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNone(result)

    def test_compute_avg_logprob_normalizes_by_token_count(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        captured = _CapturedLogits(
            sequences_scores=torch.tensor([-3.0]),
            sequences=torch.tensor([[1, 2, 3]], dtype=torch.long),
        )
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result, -1.0)

    def test_compute_avg_logprob_handles_zero_length(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        captured = _CapturedLogits(
            sequences_scores=torch.tensor([-3.0]),
            sequences=torch.zeros(1, 0, dtype=torch.long),
        )
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNone(result)

    def test_compute_avg_logprob_from_scores_when_no_sequences_scores(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        token_0_logits = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        token_2_logits = torch.tensor([[0.0, 0.0, 2.0, 0.0]])
        captured = _CapturedLogits(
            sequences_scores=None,
            sequences=torch.tensor([[0, 2]], dtype=torch.long),
            scores=(token_0_logits, token_2_logits),
        )
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNotNone(result)
        assert result is not None
        expected_0 = float(torch.log_softmax(token_0_logits.float(), dim=-1)[0, 0])
        expected_2 = float(torch.log_softmax(token_2_logits.float(), dim=-1)[0, 2])
        self.assertAlmostEqual(result, (expected_0 + expected_2) / 2, places=5)

    def test_compute_avg_logprob_prefers_sequences_scores_over_scores(self) -> None:
        from yorishiro.audio._speech_support import _CapturedLogits

        captured = _CapturedLogits(
            sequences_scores=torch.tensor([-2.0]),
            sequences=torch.tensor([[1, 2, 3]], dtype=torch.long),
            scores=(torch.tensor([[0.0]]),),
        )
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result, -2.0 / 3)

    def test_funasr_confidence_hook_captures_scores_from_greedy_generate(self) -> None:
        call_log: list[Any] = []

        class FakeLLM:
            def generate(self, *args: Any, **kwargs: Any) -> Any:
                seq = torch.tensor([[1, 2, 3]], dtype=torch.long)
                logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
                call_log.append(dict(kwargs))
                return SimpleNamespace(
                    sequences=seq,
                    sequences_scores=None,
                    scores=(logits, logits, logits),
                )

        class FakeInner:
            llm = FakeLLM()

        class FakeModel:
            model = FakeInner()

        model = FakeModel()
        captured = None
        with FunASRConfidenceHook(model) as cap:
            captured = cap
            _ = model.model.llm.generate()

        assert captured is not None
        self.assertIsNone(captured.sequences_scores)
        self.assertIsNotNone(captured.sequences)
        self.assertIsNotNone(captured.scores)
        assert captured.scores is not None
        self.assertEqual(len(captured.scores), 3)
        self.assertTrue(call_log[-1].get("output_scores", False))
        self.assertTrue(call_log[-1].get("return_dict_in_generate", False))
        result = compute_avg_logprob_from_captured_logits(captured)
        self.assertIsNotNone(result)

    def test_funasr_confidence_hook_fallback_when_no_llm(self) -> None:
        class FakeModel:
            pass

        model = FakeModel()
        captured = None
        with FunASRConfidenceHook(model) as cap:
            captured = cap

        self.assertIsNotNone(captured)
        self.assertIsNone(captured.sequences_scores)
        self.assertIsNone(captured.sequences)

    def test_qwen3_aligner_confidence_hook_fallback_when_no_thinker(self) -> None:
        class FakeAligner:
            pass

        aligner = FakeAligner()
        captured = None
        with Qwen3AlignerConfidenceHook(aligner) as cap:
            captured = cap

        self.assertIsNotNone(captured)
        self.assertEqual(len(captured.confidence_per_item), 0)


class FunASRBackendTests(unittest.TestCase):
    def test_funasr_config_instantiates_correctly(self) -> None:
        config = TranscriberConfig(
            stt_backend="funasr",
            stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        )
        self.assertEqual(config.stt_backend, "funasr")
        self.assertEqual(config.stt_model, "FunAudioLLM/Fun-ASR-MLT-Nano-2512")

    def test_funasr_worker_groups_processes_single_group(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                stt_backend="funasr",
                stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
            )
        )
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=5.0,
                end=7.5,
            ),
        ]

        class FakeFunASRModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, input, **kwargs):
                self.calls += 1
                return [
                    {
                        "key": "utt_0",
                        "text": "これはテストです。",
                        "text_tn": "これはテストです",
                        "timestamps": [
                            {
                                "token": "こ",
                                "start_time": 5.0,
                                "end_time": 5.2,
                                "score": 0.95,
                            },
                            {
                                "token": "れ",
                                "start_time": 5.2,
                                "end_time": 5.4,
                                "score": 0.90,
                            },
                            {
                                "token": "は",
                                "start_time": 5.4,
                                "end_time": 5.6,
                                "score": 0.88,
                            },
                        ],
                    }
                ]

        fake_model = FakeFunASRModel()
        saved_results: list[GroupResult] = []

        with (
            patch("yorishiro.audio.transcription.sf.read") as mock_read,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch(
                "yorishiro.audio.transcription.get_funasr_model",
                return_value=fake_model,
            ),
        ):
            mock_read.return_value = (np.zeros(40000, dtype=np.float32), 16000)
            results = transcriber._run_worker_groups_funasr(
                groups,
                worker_idx=0,
                worker_config=transcriber.config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="ja",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda result: saved_results.append(result),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].group_id, "g_000000_000000")
        self.assertGreater(len(results[0].entries), 0)
        self.assertEqual(results[0].entries[0]["text"], "これはテストです。")
        self.assertAlmostEqual(
            results[0].entries[0]["confidence"], (0.95 + 0.90 + 0.88) / 3, places=4
        )
        self.assertEqual(results[0].detected_language, "ja")
        self.assertEqual(results[0].raw_text, "これはテストです。")
        self.assertEqual(results[0].raw_alignment_text, "これはテストです")
        self.assertEqual(fake_model.calls, 1)

    def test_funasr_worker_groups_without_timestamps_uses_result_confidence(
        self,
    ) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                stt_backend="funasr",
                stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
                stt_min_confidence=0.3,
            )
        )
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=2.0,
            ),
        ]

        class FakeFunASRModel:
            def generate(self, input, **kwargs):
                return [
                    {
                        "key": "utt_0",
                        "text": "Hello world",
                        "text_tn": "Hello world",
                        "confidence": 0.42,
                    }
                ]

        with (
            patch("yorishiro.audio.transcription.sf.read") as mock_read,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch(
                "yorishiro.audio.transcription.get_funasr_model",
                return_value=FakeFunASRModel(),
            ),
        ):
            mock_read.return_value = (np.zeros(32000, dtype=np.float32), 16000)
            results = transcriber._run_worker_groups_funasr(
                groups,
                worker_idx=0,
                worker_config=transcriber.config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda _r: None,
            )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].entries[0]["confidence"], 0.42)
        self.assertAlmostEqual(results[0].entries[0]["stt_confidence"], 0.42)

    def test_funasr_worker_groups_empty_text_produces_no_entries(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                stt_backend="funasr",
                stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
            )
        )
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=1.0,
            ),
        ]

        class FakeFunASRModel:
            def generate(self, input, **kwargs):
                return [{"key": "utt_0", "text": ""}]

        with (
            patch("yorishiro.audio.transcription.sf.read") as mock_read,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch(
                "yorishiro.audio.transcription.get_funasr_model",
                return_value=FakeFunASRModel(),
            ),
        ):
            mock_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)
            results = transcriber._run_worker_groups_funasr(
                groups,
                worker_idx=0,
                worker_config=transcriber.config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language=None,
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda _r: None,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].entries), 0)

    def test_funasr_dispatches_to_funasr_worker(self) -> None:
        config = TranscriberConfig(
            stt_backend="funasr",
            stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        )
        transcriber = Transcriber(config)
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=1.0,
            ),
        ]
        with patch.object(
            transcriber, "_run_worker_groups_funasr", return_value=[]
        ) as mock_funasr:
            transcriber._run_worker_groups(
                groups=groups,
                worker_idx=0,
                worker_config=config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="ja",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda _r: None,
            )
        mock_funasr.assert_called_once()

    def test_funasr_worker_groups_uses_llm_confidence_when_no_ctc(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                stt_backend="funasr",
                stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
            )
        )
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=2.0,
            ),
        ]

        class FakeFunASRModel:
            def __init__(self) -> None:
                self.model = SimpleNamespace(llm=None)

            def generate(self, input: Any, **kwargs: Any) -> list[dict[str, Any]]:
                return [{"key": "utt_0", "text": "Hi", "text_tn": "Hi"}]

        fake_model = FakeFunASRModel()

        with (
            patch("yorishiro.audio.transcription.sf.read") as mock_read,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch(
                "yorishiro.audio.transcription.get_funasr_model",
                return_value=fake_model,
            ),
            patch(
                "yorishiro.audio.transcription.compute_avg_logprob_from_captured_logits",
                return_value=-0.25,
            ),
        ):
            mock_read.return_value = (np.zeros(32000, dtype=np.float32), 16000)
            results = transcriber._run_worker_groups_funasr(
                groups,
                worker_idx=0,
                worker_config=transcriber.config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda _r: None,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].entries), 1)
        self.assertEqual(results[0].entries[0]["text"], "Hi")
        self.assertAlmostEqual(results[0].entries[0]["confidence"], 0.7788, places=3)
        self.assertAlmostEqual(
            results[0].entries[0]["stt_confidence"], 0.7788, places=3
        )

    def test_funasr_low_confidence_filters_output(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                stt_backend="funasr",
                stt_model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
                stt_min_confidence=0.0,
            )
        )
        groups = [
            SpeechGroup(
                group_id="g_000000_000000",
                span_start_idx=0,
                span_end_idx=0,
                start=0.0,
                end=2.0,
            ),
        ]

        class FakeFunASRModel:
            def __init__(self) -> None:
                self.model = SimpleNamespace(llm=None)

            def generate(self, input: Any, **kwargs: Any) -> list[dict[str, Any]]:
                return [{"key": "utt_0", "text": "Hello", "text_tn": "Hello"}]

        fake_model = FakeFunASRModel()

        with (
            patch("yorishiro.audio.transcription.sf.read") as mock_read,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch(
                "yorishiro.audio.transcription.get_funasr_model",
                return_value=fake_model,
            ),
        ):
            mock_read.return_value = (np.zeros(32000, dtype=np.float32), 16000)
            results = transcriber._run_worker_groups_funasr(
                groups,
                worker_idx=0,
                worker_config=transcriber.config,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
                update_progress=lambda _delta: None,
                save_result=lambda _r: None,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].entries), 1)


class ForcedAlignerConfigTests(unittest.TestCase):
    def test_default_aligner_is_disabled(self) -> None:
        config = TranscriberConfig()
        self.assertFalse(config.forced_aligner_enabled)
        self.assertEqual(config.forced_aligner_backend, "qwen3")
        self.assertEqual(config.forced_aligner_model, "Qwen/Qwen3-ForcedAligner-0.6B")

    def test_aligner_config_fields(self) -> None:
        config = TranscriberConfig(
            forced_aligner_enabled=True,
            forced_aligner_backend="qwen3",
            forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_device="cuda",
            forced_aligner_min_confidence=0.3,
            forced_aligner_merge_gap_seconds=0.2,
        )
        self.assertTrue(config.forced_aligner_enabled)
        self.assertEqual(config.forced_aligner_backend, "qwen3")
        self.assertEqual(config.forced_aligner_device, "cuda")


class StripPunctuationTests(unittest.TestCase):
    def test_strips_japanese_punctuation(self) -> None:
        from yorishiro.audio.transcription import strip_punctuation_for_alignment

        self.assertEqual(
            strip_punctuation_for_alignment("こんにちは。世界！"), "こんにちは世界"
        )

    def test_strips_english_punctuation(self) -> None:
        from yorishiro.audio.transcription import strip_punctuation_for_alignment

        self.assertEqual(
            strip_punctuation_for_alignment("Hello, world! How are you?"),
            "Hello world How are you",
        )

    def test_preserves_cjk_chars(self) -> None:
        from yorishiro.audio.transcription import strip_punctuation_for_alignment

        self.assertEqual(
            strip_punctuation_for_alignment("これはテストです"), "これはテストです"
        )

    def test_empty_string(self) -> None:
        from yorishiro.audio.transcription import strip_punctuation_for_alignment

        self.assertEqual(strip_punctuation_for_alignment(""), "")


class BuildDisplayToAlignMapTests(unittest.TestCase):
    def test_basic_mapping(self) -> None:
        from yorishiro.audio.transcription import build_display_to_align_map

        display = "こんにちは。"
        align = "こんにちは"
        pos_map = build_display_to_align_map(display, align)
        assert pos_map[0] == 0  # こ
        assert pos_map[1] == 1  # ん
        assert pos_map[2] == 2  # に
        assert pos_map[3] == 3  # ち
        assert pos_map[4] == 4  # は
        assert pos_map[5] is None  # 。

    def test_english_punctuation_mapping(self) -> None:
        from yorishiro.audio.transcription import build_display_to_align_map

        display = "Hi, there!"
        align = "Hi there"
        pos_map = build_display_to_align_map(display, align)
        assert pos_map[0] == 0  # H
        assert pos_map[1] == 1  # i
        assert pos_map[2] is None  # ,
        assert pos_map[3] is None  # space
        assert pos_map[4] == 2  # t


class STTEntryAlignmentFieldsTests(unittest.TestCase):
    def test_stt_entry_accepts_optional_alignment_fields(self) -> None:
        entry = STTEntry(
            entry_id="utt_000000",
            start=0.0,
            end=1.0,
            text="こんにちは",
            confidence=0.9,
            stt_confidence=0.85,
            alignment_confidence=0.92,
        )
        self.assertEqual(entry.stt_confidence, 0.85)
        self.assertEqual(entry.alignment_confidence, 0.92)

    def test_stt_entry_defaults_alignment_fields_to_none(self) -> None:
        entry = STTEntry(
            entry_id="utt_000000",
            start=0.0,
            end=1.0,
            text="こんにちは",
            confidence=0.9,
        )
        self.assertIsNone(entry.stt_confidence)
        self.assertIsNone(entry.alignment_confidence)

    def test_stt_entry_backward_compatible_without_new_fields(self) -> None:
        entry = STTEntry(
            entry_id="utt_000000",
            start=0.0,
            end=1.0,
            text="こんにちは",
            confidence=0.9,
        )
        self.assertEqual(entry.text, "こんにちは")
        self.assertEqual(entry.confidence, 0.9)
        self.assertIsNone(entry.stt_confidence)
        self.assertIsNone(entry.alignment_confidence)


class Qwen3LanguageMappingTests(unittest.TestCase):
    def test_common_languages_map_to_names(self) -> None:
        from yorishiro.audio._speech_support import qwen3_language

        self.assertEqual(qwen3_language("ja"), "Japanese")
        self.assertEqual(qwen3_language("en"), "English")
        self.assertEqual(qwen3_language("zh"), "Chinese")
        self.assertEqual(qwen3_language("ko"), "Korean")

    def test_none_language_returns_english(self) -> None:
        from yorishiro.audio._speech_support import qwen3_language

        self.assertEqual(qwen3_language(None), "English")

    def test_unknown_language_defaults_to_english(self) -> None:
        from yorishiro.audio._speech_support import qwen3_language

        self.assertEqual(qwen3_language("xx"), "English")

    def test_normalizes_language_before_mapping(self) -> None:
        from yorishiro.audio._speech_support import qwen3_language

        self.assertEqual(qwen3_language("ja-JP"), "Japanese")
        self.assertEqual(qwen3_language("en-US"), "English")


class ForcedAlignerRegistryTests(unittest.TestCase):
    def test_aligner_config_in_cache_key_contains_aligner_fields(self) -> None:
        cfg = {
            "backend": "funasr",
            "model": "FunASRNano",
            "forced_aligner_enabled": True,
            "forced_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
        }
        parts = [
            "transcriber",
            str(cfg.get("backend", "")),
            str(cfg.get("model", "")),
            str(cfg.get("forced_aligner_enabled", "")),
            str(cfg.get("forced_aligner_model", "")),
        ]
        key = "film.audio.stt::" + "|".join(parts)
        self.assertIn("True", key)
        self.assertIn("Qwen", key)


class ForcedAlignmentFlowTests(unittest.TestCase):
    def test_align_group_result_rebases_chunk_relative_timestamps(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                forced_aligner_enabled=True,
                forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_merge_gap_seconds=0.05,
            )
        )
        result = GroupResult(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=5.0,
            end=8.0,
            entries=[
                {"text": "Hello, world!", "start": 5.0, "end": 8.0, "confidence": 0.7},
            ],
            detected_language="en",
            source_mtime=1.0,
            raw_text="Hello, world!",
            raw_alignment_text="Hello world",
        )

        captured_audio: list[object] = []

        class FakeAligner:
            def align(self, audio, text, language, **kwargs):
                del kwargs
                captured_audio.append(audio)
                self.text = text
                self.language = language
                return [
                    [
                        SimpleNamespace(text="Hello", start_time=0.2, end_time=0.8),
                        SimpleNamespace(text="world", start_time=0.9, end_time=1.5),
                    ]
                ]

        with (
            patch(
                "yorishiro.audio.transcription.get_qwen3_forced_aligner",
                return_value=FakeAligner(),
            ),
            patch(
                "yorishiro.audio.transcription.sf.read",
                return_value=(np.zeros(48000, dtype=np.float32), 16000),
            ),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
        ):
            aligned = transcriber._align_group_result(
                result,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
            )

        self.assertTrue(aligned.alignment_applied)
        self.assertEqual(len(aligned.entries), 1)
        self.assertEqual(aligned.entries[0]["text"], "Hello, world!")
        self.assertAlmostEqual(aligned.entries[0]["start"], 5.2, places=3)
        self.assertAlmostEqual(aligned.entries[0]["end"], 6.5, places=3)
        assert isinstance(captured_audio[0], tuple)
        self.assertEqual(captured_audio[0][1], 16000)

    def test_align_group_result_uses_raw_text_without_inserting_spaces(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                forced_aligner_enabled=True,
                forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
                stt_min_segment_seconds=0.05,
            )
        )
        result = GroupResult(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=0.0,
            end=2.0,
            entries=[
                {"text": "こんにちは。", "start": 0.0, "end": 1.0, "confidence": 0.7},
                {"text": "世界！", "start": 1.0, "end": 2.0, "confidence": 0.7},
            ],
            detected_language="ja",
            source_mtime=1.0,
            raw_text="こんにちは。世界！",
            raw_alignment_text="こんにちは世界",
        )

        class FakeAligner:
            def __init__(self) -> None:
                self.seen_text = ""

            def align(self, audio, text, language, **kwargs):
                del audio, language, kwargs
                self.seen_text = text
                return [
                    [
                        SimpleNamespace(
                            text="こんにちは", start_time=0.0, end_time=0.9
                        ),
                        SimpleNamespace(text="世界", start_time=1.0, end_time=1.8),
                    ]
                ]

        fake_aligner = FakeAligner()
        with (
            patch(
                "yorishiro.audio.transcription.get_qwen3_forced_aligner",
                return_value=fake_aligner,
            ),
            patch(
                "yorishiro.audio.transcription.sf.read",
                return_value=(np.zeros(32000, dtype=np.float32), 16000),
            ),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
        ):
            aligned = transcriber._align_group_result(
                result,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="ja",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
            )

        self.assertEqual(fake_aligner.seen_text, "こんにちは世界")
        self.assertEqual(aligned.raw_text, "こんにちは。世界！")

    def test_align_group_result_splits_on_alignment_pause_without_punctuation(
        self,
    ) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                forced_aligner_enabled=True,
                forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
                stt_min_segment_seconds=0.05,
            )
        )
        result = GroupResult(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=2.0,
            end=4.0,
            entries=[
                {"text": "Hello there", "start": 2.0, "end": 4.0, "confidence": 0.7},
            ],
            detected_language="en",
            source_mtime=1.0,
            raw_text="Hello there",
            raw_alignment_text="Hello there",
        )

        class FakeAligner:
            def align(self, audio, text, language, **kwargs):
                del audio, text, language, kwargs
                return [
                    [
                        SimpleNamespace(text="Hello", start_time=0.0, end_time=0.2),
                        SimpleNamespace(text="there", start_time=0.95, end_time=1.2),
                    ]
                ]

        with (
            patch(
                "yorishiro.audio.transcription.get_qwen3_forced_aligner",
                return_value=FakeAligner(),
            ),
            patch(
                "yorishiro.audio.transcription.sf.read",
                return_value=(np.zeros(32000, dtype=np.float32), 16000),
            ),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
        ):
            aligned = transcriber._align_group_result(
                result,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
            )

        self.assertTrue(aligned.alignment_applied)
        self.assertEqual([entry["text"] for entry in aligned.entries], ["Hello there"])
        self.assertAlmostEqual(aligned.entries[0]["start"], 2.0)
        self.assertAlmostEqual(aligned.entries[0]["end"], 3.2)

    def test_align_group_result_rejects_zero_duration_aligned_entries(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                forced_aligner_enabled=True,
                forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
            )
        )
        result = GroupResult(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=10.0,
            end=12.0,
            entries=[
                {
                    "text": "あっ、あっ、あっ",
                    "start": 10.0,
                    "end": 12.0,
                    "confidence": 0.0,
                },
            ],
            detected_language="ja",
            source_mtime=1.0,
            raw_text="あっ、あっ、あっ",
            raw_alignment_text="あっあっあっ",
        )

        class FakeAligner:
            def align(self, audio, text, language, **kwargs):
                del audio, text, language, kwargs
                return [
                    [
                        SimpleNamespace(
                            text="あっあっあっ", start_time=0.0, end_time=0.0
                        ),
                    ]
                ]

        with (
            patch(
                "yorishiro.audio.transcription.get_qwen3_forced_aligner",
                return_value=FakeAligner(),
            ),
            patch(
                "yorishiro.audio.transcription.sf.read",
                return_value=(np.zeros(32000, dtype=np.float32), 16000),
            ),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
        ):
            aligned = transcriber._align_group_result(
                result,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="ja",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
            )

        self.assertFalse(aligned.alignment_applied)
        self.assertEqual(aligned.entries, result.entries)

    def test_align_group_result_whisper_backend(self) -> None:
        transcriber = Transcriber(
            TranscriberConfig(
                forced_aligner_enabled=True,
                forced_aligner_backend="whisper",
            )
        )
        result = GroupResult(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=2.0,
            end=4.0,
            entries=[
                {"text": "Hello there", "start": 2.0, "end": 4.0, "confidence": 0.7},
            ],
            detected_language="en",
            source_mtime=1.0,
            raw_text="Hello there",
            raw_alignment_text="Hello there",
        )

        class FakeWord:
            def __init__(self, word, start, end, probability):
                self.word = word
                self.start = start
                self.end = end
                self.probability = probability

        class FakeSegment:
            def __init__(self, words):
                self.words = words

        class FakeInfo:
            pass

        class FakeWhisperModel:
            def transcribe(self, audio_path, **kwargs):
                return iter(
                    [
                        FakeSegment(
                            [
                                FakeWord("Hello", 0.0, 0.8, 0.9),
                                FakeWord("there", 0.8, 1.6, 0.85),
                            ]
                        )
                    ]
                ), FakeInfo()

        with (
            patch(
                "yorishiro.audio.transcription.get_whisper_forced_aligner",
                return_value=FakeWhisperModel(),
            ),
            patch(
                "yorishiro.audio.transcription.sf.read",
                return_value=(np.zeros(32000, dtype=np.float32), 16000),
            ),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
        ):
            aligned = transcriber._align_group_result(
                result,
                audio_path=Path("/tmp/test.wav"),
                file_sample_rate=16000,
                language="en",
                resample_module=types.SimpleNamespace(
                    resample=lambda audio, **_kw: audio
                ),
            )

        self.assertTrue(aligned.alignment_applied)

    def test_run_transcription_aligns_when_resuming_from_checkpoints(self) -> None:
        transcriber = Transcriber(TranscriberConfig(forced_aligner_enabled=True))
        group = SpeechGroup(
            group_id="g_000000_000000",
            span_start_idx=0,
            span_end_idx=0,
            start=0.0,
            end=1.0,
        )
        completed = {
            group.group_id: GroupResult(
                group_id=group.group_id,
                span_start_idx=group.span_start_idx,
                span_end_idx=group.span_end_idx,
                start=group.start,
                end=group.end,
                entries=[{"text": "a", "start": 0.0, "end": 1.0, "confidence": 0.5}],
                detected_language="en",
                source_mtime=1.0,
            )
        }

        with (
            patch("yorishiro.audio.transcription.sf.SoundFile") as FakeSoundFile,
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_mtime=1.0)),
            patch.object(
                transcriber,
                "_speech_spans",
                return_value=[SpeechSpan(index=0, start=0.0, end=1.0)],
            ),
            patch.object(transcriber, "_build_speech_groups", return_value=[group]),
            patch.object(
                transcriber,
                "_prepare_checkpoint_dir",
                return_value=Path("/tmp/.stt_checkpoints"),
            ),
            patch.object(transcriber, "_load_group_results", return_value=completed),
            patch.object(
                transcriber,
                "_maybe_align_results",
                wraps=transcriber._maybe_align_results,
            ) as maybe_align,
            patch.object(
                transcriber,
                "_assemble_transcript",
                return_value=STTTranscript(language="en", entries=[]),
            ),
        ):
            FakeSoundFile.return_value.__enter__ = lambda s: SimpleNamespace(
                samplerate=16000, frames=16000
            )
            FakeSoundFile.return_value.__exit__ = lambda s, *a: None
            with patch.object(
                transcriber,
                "_align_group_result",
                side_effect=lambda result, **kwargs: result,
            ):
                transcriber._run_transcription(
                    Path("/tmp/test.wav"),
                    [{"start": 0.0, "end": 1.0}],
                    "en",
                    output_dir=Path("/tmp"),
                    input_signature="sig",
                )

        maybe_align.assert_called_once()


class SpeakerAttributorTests(unittest.TestCase):
    def test_unknown_attribution_preserves_stt_entry_ids(self) -> None:
        attributor = SpeakerAttributor()
        stt = STTTranscript(
            language="ja",
            entries=[
                STTEntry(
                    entry_id="utt_custom_001",
                    start=0.0,
                    end=1.0,
                    text="a",
                    confidence=0.9,
                ),
                STTEntry(
                    entry_id="utt_custom_002",
                    start=1.0,
                    end=2.0,
                    text="b",
                    confidence=0.8,
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            bank = SpeakerBankManager()
            result = attributor._unknown_attribution(stt, Path(tmp_dir), bank)
        self.assertEqual(result.entries[0].entry_id, "utt_custom_001")
        self.assertEqual(result.entries[1].entry_id, "utt_custom_002")

    def test_default_config_uses_utterance_averaged_umap_hdbscan(self) -> None:
        cfg = SpeakerAttributorConfig()
        self.assertEqual(cfg.clustering_method, "umap_hdbscan_auto")
        self.assertEqual(cfg.utterance_aggregation, "medoid")
        self.assertEqual(cfg.umap_n_neighbors, 5)
        self.assertEqual(cfg.umap_n_components, 5)
        self.assertEqual(cfg.hdbscan_min_cluster_size, 10)
        self.assertFalse(cfg.diagnostics_enabled)

    def test_utterance_averaged_clustering_expands_labels_to_windows(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(clustering_method="umap_hdbscan_manual")
        )

        X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=1, start=3.0, end=4.0),
            EmbeddingWindow(stt_idx=1, start=4.0, end=5.0),
        ]
        valid_indices = [0, 1, 2, 3]

        def fake_run(
            X_in: np.ndarray,
            *,
            n_neighbors: int,
            min_cluster_size: int,
            n_components: int,
        ) -> tuple[np.ndarray, float]:
            del n_neighbors, min_cluster_size, n_components
            # Should be clustering 2 utterances (averaged embeddings), not 4 windows.
            self.assertEqual(X_in.shape[0], 2)
            norms = np.linalg.norm(X_in, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))
            return np.array([0, 1], dtype=np.int64), 0.8

        with patch.object(attributor, "_run_umap_hdbscan", side_effect=fake_run):
            labels = attributor._cluster_windows(X, windows, valid_indices)

        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def test_utterance_averaged_clustering_uses_post_filter_set(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(clustering_method="umap_hdbscan_manual")
        )

        # Only these two windows are considered (post-filter set).
        X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=1, start=3.0, end=4.0),
            EmbeddingWindow(stt_idx=1, start=4.0, end=5.0),
        ]
        valid_indices = [0, 2]

        def fake_run(
            X_in: np.ndarray,
            *,
            n_neighbors: int,
            min_cluster_size: int,
            n_components: int,
        ) -> tuple[np.ndarray, float]:
            del n_neighbors, min_cluster_size, n_components
            self.assertEqual(X_in.shape, (2, 2))
            # With one window per utterance in the post-filter set, means equal those windows.
            self.assertTrue(
                np.allclose(X_in[0], np.array([1.0, 0.0], dtype=np.float32))
            )
            self.assertTrue(
                np.allclose(X_in[1], np.array([0.0, 1.0], dtype=np.float32))
            )
            return np.array([0, 1], dtype=np.int64), 0.7

        with patch.object(attributor, "_run_umap_hdbscan", side_effect=fake_run):
            labels = attributor._cluster_windows(X, windows, valid_indices)

        self.assertEqual(labels.tolist(), [0, 1])

    def test_build_utterance_embeddings_uses_medoid_not_mean(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(utterance_aggregation="medoid")
        )
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.8, 0.2], dtype=np.float32)
        emb_c = np.array([0.0, 1.0], dtype=np.float32)
        X = np.stack([emb_a, emb_b, emb_c])
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=0, start=2.0, end=3.0),
        ]
        valid_indices = [0, 1, 2]

        seg_ids, utterance_embs = attributor._build_utterance_embeddings(
            X, windows, valid_indices
        )

        self.assertEqual(seg_ids, [0])
        expected = emb_b / np.linalg.norm(emb_b)
        self.assertTrue(np.allclose(utterance_embs[0], expected, atol=1e-6))

    def test_build_utterance_embeddings_can_use_mean(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(utterance_aggregation="mean")
        )
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.8, 0.2], dtype=np.float32)
        emb_c = np.array([0.0, 1.0], dtype=np.float32)
        X = np.stack([emb_a, emb_b, emb_c])
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=0, start=2.0, end=3.0),
        ]
        valid_indices = [0, 1, 2]

        seg_ids, utterance_embs = attributor._build_utterance_embeddings(
            X, windows, valid_indices
        )

        self.assertEqual(seg_ids, [0])
        expected = np.mean(X, axis=0)
        expected = expected / np.linalg.norm(expected)
        self.assertTrue(np.allclose(utterance_embs[0], expected, atol=1e-6))

    def test_medoid_with_two_windows_picks_first_on_tie(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(utterance_aggregation="medoid")
        )
        X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
        ]

        seg_ids, utterance_embs = attributor._build_utterance_embeddings(
            X, windows, [0, 1]
        )

        self.assertEqual(seg_ids, [0])
        self.assertTrue(
            np.allclose(utterance_embs[0], np.array([1.0, 0.0], dtype=np.float32))
        )

    def test_vote_consistency_metrics_report_low_disagreement_for_clean_case(
        self,
    ) -> None:
        attributor = SpeakerAttributor(SpeakerAttributorConfig())
        window_X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.95, 0.05], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.05, 0.95], dtype=np.float32),
            ]
        )
        window_X = window_X / np.maximum(
            np.linalg.norm(window_X, axis=1, keepdims=True), 1e-10
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=1, start=3.0, end=4.0),
            EmbeddingWindow(stt_idx=1, start=4.0, end=5.0),
        ]
        disagreement_rate, voted_separation = attributor._vote_consistency_metrics(
            window_X,
            windows,
            [0, 1, 2, 3],
            [0, 1],
            np.array([0, 1], dtype=np.int64),
        )

        self.assertAlmostEqual(disagreement_rate, 0.0, places=6)
        self.assertGreater(voted_separation, 0.5)

    def test_evaluate_clustering_reports_expected_metrics(self) -> None:
        attributor = SpeakerAttributor(SpeakerAttributorConfig())
        utterance_X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        )
        window_X = np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.95, 0.05], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.05, 0.95], dtype=np.float32),
            ]
        )
        window_X = window_X / np.maximum(
            np.linalg.norm(window_X, axis=1, keepdims=True), 1e-10
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=1.0),
            EmbeddingWindow(stt_idx=0, start=1.0, end=2.0),
            EmbeddingWindow(stt_idx=1, start=3.0, end=4.0),
            EmbeddingWindow(stt_idx=1, start=4.0, end=5.0),
        ]
        breakdown = attributor._evaluate_clustering(
            utterance_X,
            np.array([0, 1], dtype=np.int64),
            window_X,
            windows,
            [0, 1, 2, 3],
            [0, 1],
            mean_persistence=0.5,
        )

        self.assertIsInstance(breakdown, ClusteringScoreBreakdown)
        self.assertAlmostEqual(breakdown.disagreement_rate, 0.0, places=6)
        self.assertGreater(breakdown.voted_separation, 0.5)

    def test_score_usefulness_rank_helper(self) -> None:
        rows: list[SweepCandidateRow] = [
            {
                "nn": 5,
                "dim": 5,
                "mcs": 2,
                "score": 1.0,
                "clusters": 10,
                "n_noise": 1,
                "top_str": "",
                "labels": np.array([0, 0], dtype=np.int64),
                "nn_purity": 0.8,
                "disagree": 0.2,
                "sep": 0.3,
                "silhouette": 0.1,
                "singleton": 0.1,
                "noise": 0.1,
                "coarse": 0.8,
                "persistence": 0.5,
            },
            {
                "nn": 10,
                "dim": 5,
                "mcs": 3,
                "score": 2.0,
                "clusters": 8,
                "n_noise": 0,
                "top_str": "",
                "labels": np.array([0, 1], dtype=np.int64),
                "nn_purity": 0.7,
                "disagree": 0.1,
                "sep": 0.4,
                "silhouette": 0.2,
                "singleton": 0.2,
                "noise": 0.0,
                "coarse": 0.7,
                "persistence": 0.6,
            },
            {
                "nn": 15,
                "dim": 10,
                "mcs": 5,
                "score": 1.5,
                "clusters": 6,
                "n_noise": 2,
                "top_str": "",
                "labels": np.array([1, 1], dtype=np.int64),
                "nn_purity": 0.9,
                "disagree": 0.3,
                "sep": 0.2,
                "silhouette": 0.05,
                "singleton": 0.3,
                "noise": 0.2,
                "coarse": 0.9,
                "persistence": 0.4,
            },
        ]

        selected = rows[1]
        self.assertEqual(
            SpeakerAttributor._metric_rank(rows, selected, "nn_purity", reverse=True),
            3,
        )
        self.assertEqual(
            SpeakerAttributor._metric_rank(rows, selected, "disagree", reverse=False),
            1,
        )

    def test_embed_windows_uses_embedding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "voice.flac"
            output_dir = tmp_path / "audio"
            output_dir.mkdir()

            samples = np.linspace(-0.25, 0.25, 32000, dtype=np.float32)
            import soundfile as sf

            sf.write(audio_path, samples, 16000)

            attributor = SpeakerAttributor(SpeakerAttributorConfig())
            windows = [EmbeddingWindow(stt_idx=0, start=0.0, end=1.0)]
            bank = SpeakerBankManager()
            emb = np.array([1.0, 0.0], dtype=np.float32)
            stt = STTTranscript(
                language="ja",
                entries=[STTEntry(start=0.0, end=1.2, text="a", confidence=0.9)],
            )

            with (
                patch.object(attributor, "_window_has_energy", return_value=True),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    return_value=emb,
                ) as extract,
            ):
                valid_1, embs_1 = attributor._embed_windows(
                    windows, audio_path, bank, output_dir, stt=stt
                )
                valid_2, embs_2 = attributor._embed_windows(
                    windows, audio_path, bank, output_dir, stt=stt
                )

            self.assertEqual(extract.call_count, 1)
            self.assertEqual(valid_1, [0])
            self.assertEqual(valid_2, [0])
            self.assertTrue(np.array_equal(embs_1[0], emb))
            self.assertTrue(np.array_equal(embs_2[0], emb))
            self.assertTrue((output_dir / "speaker_embedding_cache.npz").exists())

    def test_embed_windows_force_regenerates_embedding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "voice.flac"
            output_dir = tmp_path / "audio"
            output_dir.mkdir()

            samples = np.linspace(-0.25, 0.25, 32000, dtype=np.float32)
            import soundfile as sf

            sf.write(audio_path, samples, 16000)

            attributor = SpeakerAttributor(SpeakerAttributorConfig())
            windows = [EmbeddingWindow(stt_idx=0, start=0.0, end=1.0)]
            bank = SpeakerBankManager()
            emb = np.array([1.0, 0.0], dtype=np.float32)
            stt = STTTranscript(
                language="ja",
                entries=[STTEntry(start=0.0, end=1.2, text="a", confidence=0.9)],
            )

            with (
                patch.object(attributor, "_window_has_energy", return_value=True),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    return_value=emb,
                ) as extract,
            ):
                attributor._embed_windows(
                    windows, audio_path, bank, output_dir, stt=stt
                )
                attributor._embed_windows(
                    windows,
                    audio_path,
                    bank,
                    output_dir,
                    stt=stt,
                    force=True,
                )

            self.assertEqual(extract.call_count, 2)

    def test_embed_windows_cache_invalidates_when_stt_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            audio_path = tmp_path / "voice.flac"
            output_dir = tmp_path / "audio"
            output_dir.mkdir()

            samples = np.linspace(-0.25, 0.25, 32000, dtype=np.float32)
            import soundfile as sf

            sf.write(audio_path, samples, 16000)

            attributor = SpeakerAttributor(SpeakerAttributorConfig())
            windows = [EmbeddingWindow(stt_idx=0, start=0.0, end=1.0)]
            bank = SpeakerBankManager()
            emb = np.array([1.0, 0.0], dtype=np.float32)
            stt_a = STTTranscript(
                language="ja",
                entries=[STTEntry(start=0.0, end=1.2, text="a", confidence=0.9)],
            )
            stt_b = STTTranscript(
                language="ja",
                entries=[STTEntry(start=0.0, end=1.3, text="a", confidence=0.9)],
            )

            with (
                patch.object(attributor, "_window_has_energy", return_value=True),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    return_value=emb,
                ) as extract,
            ):
                attributor._embed_windows(
                    windows, audio_path, bank, output_dir, stt=stt_a
                )
                attributor._embed_windows(
                    windows, audio_path, bank, output_dir, stt=stt_b
                )

            self.assertEqual(extract.call_count, 2)

    def test_windowed_clustering_merges_similar_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            stt = STTTranscript(
                language="ja",
                entries=[
                    STTEntry(start=0.0, end=2.0, text="a", confidence=0.1),
                    STTEntry(start=3.0, end=5.0, text="b", confidence=0.2),
                ],
            )
            (output_dir / "stt.json").write_text(
                stt.model_dump_json(indent=2), encoding="utf-8"
            )

            emb_a1 = np.array([1.0, 0.0], dtype=np.float32)
            emb_a2 = np.array([1.01, 0.01], dtype=np.float32)
            emb_b1 = np.array([0.99, -0.01], dtype=np.float32)
            emb_b2 = np.array([1.0, 0.02], dtype=np.float32)
            attributor = SpeakerAttributor(
                SpeakerAttributorConfig(clustering_method="umap_hdbscan_manual")
            )
            with (
                patch.object(attributor, "_window_has_energy", return_value=True),
                patch.object(
                    attributor,
                    "_run_umap_hdbscan",
                    return_value=(np.array([0, 0], dtype=np.int64), 0.8),
                ),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    side_effect=[emb_a1, emb_a2, emb_b1, emb_b2],
                ),
            ):
                result = attributor.run(audio_path, output_dir)

            self.assertEqual(result.entries[0].speaker_id, result.entries[1].speaker_id)
            self.assertTrue((output_dir / "speaker_bank.json").exists())

    def test_windowed_clustering_separates_different_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            stt = STTTranscript(
                language="ja",
                entries=[
                    STTEntry(start=0.0, end=2.0, text="a", confidence=0.1),
                    STTEntry(start=3.0, end=5.0, text="b", confidence=0.2),
                ],
            )
            (output_dir / "stt.json").write_text(
                stt.model_dump_json(indent=2), encoding="utf-8"
            )

            emb1 = np.array([1.0, 0.0], dtype=np.float32)
            emb2 = np.array([1.01, 0.01], dtype=np.float32)
            emb3 = np.array([0.0, 1.0], dtype=np.float32)
            emb4 = np.array([0.01, 0.99], dtype=np.float32)
            attributor = SpeakerAttributor(
                SpeakerAttributorConfig(clustering_method="umap_hdbscan_manual")
            )
            with (
                patch.object(attributor, "_window_has_energy", return_value=True),
                patch.object(
                    attributor,
                    "_run_umap_hdbscan",
                    return_value=(np.array([0, 1], dtype=np.int64), 0.8),
                ),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    side_effect=[emb1, emb2, emb3, emb4],
                ),
            ):
                result = attributor.run(audio_path, output_dir)

            self.assertNotEqual(
                result.entries[0].speaker_id, result.entries[1].speaker_id
            )
            self.assertEqual(len({e.speaker_id for e in result.entries}), 2)

    def test_single_window_low_similarity_falls_back_to_time(self) -> None:
        attributor = SpeakerAttributor(SpeakerAttributorConfig(min_vote_similarity=0.2))
        stt = STTTranscript(
            language="ja",
            entries=[
                STTEntry(start=0.0, end=0.3, text="!", confidence=0.0),
                STTEntry(start=0.5, end=1.5, text="a", confidence=0.1),
            ],
        )
        windows = [
            EmbeddingWindow(stt_idx=0, start=0.0, end=0.3),
            EmbeddingWindow(stt_idx=1, start=0.5, end=1.5),
        ]
        valid_indices_all = [0, 1]
        embeddings_all = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]
        speaker_centroids = {"SPKR_001": np.array([1.0, 0.0], dtype=np.float32)}

        result_entries = attributor._vote_speakers(
            stt=stt,
            windows=windows,
            valid_indices_all=valid_indices_all,
            embeddings_all=embeddings_all,
            speaker_centroids=speaker_centroids,
        )

        self.assertEqual(result_entries[1].speaker_id, "SPKR_001")
        self.assertEqual(result_entries[0].speaker_id, "SPKR_001")
        self.assertTrue(result_entries[0].embedding_present)
        self.assertAlmostEqual(float(result_entries[0].similarity or 0.0), 0.0)

    def test_extract_windows_produces_correct_windows(self) -> None:
        attributor = SpeakerAttributor(
            SpeakerAttributorConfig(
                window_duration=1.5,
                window_hop=0.75,
                min_window_duration=0.8,
            )
        )
        stt = STTTranscript(
            language="ja",
            entries=[
                STTEntry(start=0.0, end=0.5, text="short", confidence=0.1),
                STTEntry(start=1.0, end=2.0, text="medium", confidence=0.2),
                STTEntry(start=3.0, end=6.0, text="long", confidence=0.3),
            ],
        )
        windows = attributor._extract_windows(stt)
        stt_indices = [w.stt_idx for w in windows]
        self.assertIn(0, stt_indices)
        self.assertIn(1, stt_indices)
        self.assertIn(2, stt_indices)

        very_short_wins = [w for w in windows if w.stt_idx == 0]
        self.assertEqual(len(very_short_wins), 1)
        self.assertAlmostEqual(very_short_wins[0].start, 0.0)
        self.assertAlmostEqual(very_short_wins[0].end, 0.5)

        short_wins = [w for w in windows if w.stt_idx == 1]
        self.assertEqual(len(short_wins), 1)
        self.assertAlmostEqual(short_wins[0].start, 1.0)
        self.assertAlmostEqual(short_wins[0].end, 2.0)
        long_wins = [w for w in windows if w.stt_idx == 2]
        self.assertGreaterEqual(len(long_wins), 3)

    def test_energy_gating_skips_silent_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "voice.flac"
            audio_path.write_bytes(b"stub")
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir()
            stt = STTTranscript(
                language="ja",
                entries=[
                    STTEntry(start=0.0, end=2.0, text="a", confidence=0.1),
                ],
            )
            (output_dir / "stt.json").write_text(
                stt.model_dump_json(indent=2), encoding="utf-8"
            )

            call_count = 0

            def fake_extract(_path: Path, _start: float, _end: float) -> np.ndarray:
                nonlocal call_count
                call_count += 1
                return np.array([1.0, 0.0], dtype=np.float32)

            attributor = SpeakerAttributor(
                SpeakerAttributorConfig(clustering_method="umap_hdbscan_auto")
            )
            with (
                patch.object(attributor, "_window_has_energy", return_value=False),
                patch(
                    "yorishiro.audio.speaker_attribution.SpeakerBankManager.extract_speaker_embedding",
                    side_effect=fake_extract,
                ),
            ):
                result = attributor.run(audio_path, output_dir)

            self.assertEqual(call_count, 0)
            self.assertEqual(result.entries[0].speaker_id, "UNKNOWN")


class AudioSeparatorTests(unittest.TestCase):
    def test_stitch_chunk_first_chunk_splits_body_and_tail(self) -> None:
        current = np.vstack(
            [
                np.arange(10, dtype=np.float32),
                np.arange(10, dtype=np.float32),
            ]
        )
        write_block, next_tail = AudioSeparator._stitch_chunk(
            np.zeros((2, 0), dtype=np.float32),
            current,
            overlap_samples=3,
        )
        self.assertEqual(write_block.shape, (2, 7))
        self.assertEqual(next_tail.shape, (2, 3))
        self.assertTrue(np.allclose(write_block, current[:, :7]))
        self.assertTrue(np.allclose(next_tail, current[:, 7:]))

    def test_stitch_chunk_crossfades_pending_and_current(self) -> None:
        pending = np.full((2, 3), 1.0, dtype=np.float32)
        current = np.full((2, 6), 3.0, dtype=np.float32)
        write_block, next_tail = AudioSeparator._stitch_chunk(
            pending, current, overlap_samples=3
        )
        # For full overlap, blended center should be [1,2,3] per channel.
        expected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.assertTrue(np.allclose(write_block[0:1, :3], expected, atol=1e-6))
        self.assertEqual(next_tail.shape, (2, 3))
        self.assertTrue(np.allclose(next_tail, np.full((2, 3), 3.0, dtype=np.float32)))

    def test_stitch_chunk_without_overlap_concatenates(self) -> None:
        pending = np.ones((2, 2), dtype=np.float32)
        current = np.full((2, 4), 2.0, dtype=np.float32)
        write_block, next_tail = AudioSeparator._stitch_chunk(
            pending, current, overlap_samples=0
        )
        self.assertEqual(write_block.shape, (2, 6))
        self.assertEqual(next_tail.shape, (2, 0))
        self.assertTrue(np.allclose(write_block[:, :2], 1.0))
        self.assertTrue(np.allclose(write_block[:, 2:], 2.0))


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
            (output_dir / "stt.json").write_text(
                stt.model_dump_json(indent=2), encoding="utf-8"
            )
            (output_dir / "speaker_attribution.json").write_text(
                attribution.model_dump_json(indent=2), encoding="utf-8"
            )

            analyzer = EmotionAnalyzer()

            def fake_analyze_emotions(
                _audio_path: Path, transcript_in: Transcript
            ) -> Transcript:
                transcript_in.entries[0].emotion = "happy"
                return transcript_in

            def fake_analyze_prosody(
                _audio_path: Path, transcript_in: Transcript
            ) -> Transcript:
                transcript_in.entries[0].volume = "normal"
                return transcript_in

            with (
                patch.object(
                    analyzer, "_analyze_emotions", side_effect=fake_analyze_emotions
                ),
                patch.object(
                    analyzer, "_analyze_prosody", side_effect=fake_analyze_prosody
                ),
            ):
                result = analyzer.run(audio_path, output_dir)

            self.assertEqual(result.entries[0].emotion, "happy")
            saved = json.loads(
                (output_dir / "transcript.json").read_text(encoding="utf-8")
            )
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
            speakers_task = FilmAudioSpeakersStep(
                project, "film-src", registry
            ).tasks()[0]
            emotion_task = FilmAudioEmotionStep(project, "film-src", registry).tasks()[
                0
            ]

            self.assertIsInstance(vad_task, FilmAudioVADTask)
            self.assertIsInstance(stt_task, FilmAudioSTTTask)
            self.assertIsInstance(speakers_task, FilmAudioSpeakersTask)
            self.assertIsInstance(emotion_task, FilmAudioEmotionTask)
            assert isinstance(stt_task, FilmAudioSTTTask)
            self.assertEqual(stt_task._language, "ja")
            self.assertEqual(
                vad_task.output_paths(),
                [project.step_dir("film-src", "audio") / "vad.json"],
            )
            self.assertEqual(
                vad_task.input_paths(),
                [
                    project.step_dir("film-src", "audio") / "voice.flac",
                    project.step_dir("film-src", "audio") / "nonvoice.flac",
                ],
            )
            self.assertEqual(
                speakers_task.output_paths(),
                [
                    project.step_dir("film-src", "audio") / "speaker_attribution.json",
                    project.step_dir("film-src", "audio") / "speaker_bank.json",
                    project.step_dir("film-src", "audio") / "speaker_embeddings.pkl",
                    project.step_dir("film-src", "audio")
                    / "speaker_embedding_cache.npz",
                ],
            )
            self.assertEqual(
                emotion_task.output_paths(),
                [project.step_dir("film-src", "audio") / "transcript.json"],
            )

    def test_speakers_task_forwards_force_to_attributor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            attributor = Mock()
            runtime = cast(StepRuntime, SimpleNamespace(instance=lambda: attributor))
            task = FilmAudioSpeakersTask(output_dir, runtime)

            task.run(force=True)

            attributor.run.assert_called_once_with(
                output_dir / "voice.flac",
                output_dir,
                force=True,
            )
