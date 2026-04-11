from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np

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
    GroupResult,
    SpeechGroup,
    Transcriber,
    TranscriberConfig,
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
