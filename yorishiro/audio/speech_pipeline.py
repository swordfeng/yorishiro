"""Speech transcription, speaker diarization, and emotion analysis.

Pipeline:
1. VAD preprocessing (Silero-VAD)
2. Speaker diarization (pyannote-audio)
3. Speech-to-text (faster-whisper)
4. Emotion/tonality analysis (emotion2vec)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import soundfile as sf
import torch
from av.audio.frame import AudioFrame
from pydantic import BaseModel, Field

from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.utils import get_device


class SpeechPipelineConfig(BaseModel):
    vad_backend: str = Field(default="silero-vad", description="VAD backend")
    diarization_backend: str = Field(default="pyannote", description="Diarization backend")
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    stt_backend: str = Field(default="faster-whisper", description="STT backend")
    stt_model: str = Field(default="large-v3", description="Whisper model size")
    language: str | None = Field(default=None, description="Language hint (None for auto-detect)")
    emotion_backend: str = Field(default="emotion2vec", description="Emotion detection backend")
    hf_token_env: str = Field(default="HF_TOKEN", description="HuggingFace token env var")


class SpeechPipeline:
    """Full speech processing: VAD, diarization, STT, emotion."""

    def __init__(self, config: SpeechPipelineConfig | None = None):
        self.config = config or SpeechPipelineConfig()
        self._diarization_pipeline = None
        self._whisper_model = None
        self._emotion_model = None

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        language: str | None = None,
        force: bool = False,
    ) -> Transcript:
        """Process audio from video and return transcript.

        Caches to output_dir / transcript.json.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_file = output_dir / "transcript.json"

        if not force and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return Transcript(**cached)
            except Exception:
                pass

        print(f"  [SpeechPipeline] Extracting audio from {video_path.name} ...")
        audio_path = self._extract_audio(video_path, output_dir)

        print("  [SpeechPipeline] Running VAD ...")
        speech_segments = self._run_vad(audio_path)

        print("  [SpeechPipeline] Running speaker diarization ...")
        diarization = self._run_diarization(audio_path)

        print("  [SpeechPipeline] Running transcription ...")
        detected_language = language or self.config.language
        transcript = self._run_transcription(audio_path, diarization, speech_segments, detected_language)

        print("  [SpeechPipeline] Running emotion analysis ...")
        transcript = self._analyze_emotions(audio_path, transcript)

        cache_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [SpeechPipeline] Transcribed {len(transcript.entries)} segments")
        return transcript

    def _extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Extract audio from video as WAV using PyAV."""
        audio_path = output_dir / "audio.wav"

        if audio_path.exists():
            return audio_path

        input_container = av.open(str(video_path))
        audio_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            raise ValueError(f"No audio stream found in {video_path}")

        output_container = av.open(str(audio_path), "w")
        output_stream = output_container.add_stream("pcm_s16le", rate=16000)
        output_stream.channels = 1

        for frame in input_container.decode(audio_stream):
            assert isinstance(frame, AudioFrame)
            if frame.sample_rate != 16000:
                frame = frame.resample(16000)  # ty:ignore[unresolved-attribute]
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()

        return audio_path

    def _run_vad(self, audio_path: Path) -> list[dict]:
        """Run Voice Activity Detection."""
        try:
            from silero_vad import load_silero_vad, read_audio  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
        except ImportError:
            print("    [VAD] silero-vad not installed, skipping VAD")
            return [{"start": 0.0, "end": float("inf")}]

        model = load_silero_vad()
        wav = read_audio(str(audio_path))

        speech_segments = []
        for i in range(0, len(wav), 16000):
            chunk = wav[i:i + 16000]
            if len(chunk) < 16000:
                chunk = torch.nn.functional.pad(chunk, (0, 16000 - len(chunk)))

            speech_prob = model(chunk, 16000)
            if speech_prob > 0.5:
                speech_segments.append({
                    "start": i / 16000.0,
                    "end": min((i + 16000) / 16000.0, len(wav) / 16000.0),
                })

        return speech_segments or [{"start": 0.0, "end": float("inf")}]

    def _run_diarization(self, audio_path: Path) -> list[dict]:
        """Run speaker diarization."""
        try:
            from pyannote.audio import Pipeline

            hf_token = os.environ.get(self.config.hf_token_env)
            if not hf_token:
                print(f"    [Diarization] {self.config.hf_token_env} not set, using single-speaker fallback")
                return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

            if self._diarization_pipeline is None:
                pipeline = Pipeline.from_pretrained(
                    self.config.diarization_model,
                    token=hf_token,
                )
                device = torch.device(get_device())
                pipeline = pipeline.to(device)  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
                self._diarization_pipeline = pipeline

            assert self._diarization_pipeline is not None
            diarization = self._diarization_pipeline(str(audio_path))

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                })

            return segments if segments else [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

        except Exception as e:
            print(f"    [Diarization] Error: {e}, using single-speaker fallback")
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

    def _run_transcription(
        self,
        audio_path: Path,
        diarization: list[dict],
        speech_segments: list[dict],
        language: str | None,
    ) -> Transcript:
        """Run speech-to-text on the full audio, then assign speakers by overlap."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("    [STT] faster-whisper not installed, using dummy transcript")
            entries = [
                TranscriptEntry(
                    speaker_global="SPEAKER_00",
                    start=seg["start"] if seg["start"] != float("inf") else 0.0,
                    end=seg["end"] if seg["end"] != float("inf") else 0.0,
                    text="[Transcription unavailable]",
                    confidence=0.0,
                )
                for seg in diarization
                if seg["start"] != float("inf")
            ]
            return Transcript(language=language or "unknown", entries=entries)

        if self._whisper_model is None:
            # faster-whisper uses CTranslate2 which only supports CUDA/CPU, not MPS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._whisper_model = WhisperModel(self.config.stt_model, device=device)

        detected_language = language or "auto"

        # Transcribe the full audio once
        segments, info = self._whisper_model.transcribe(
            str(audio_path),
            language=detected_language if detected_language != "auto" else None,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        entries = []
        for segment in segments:
            speaker = self._assign_speaker(segment.start, segment.end, diarization)
            entries.append(TranscriptEntry(
                speaker_global=speaker,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=segment.avg_logprob if hasattr(segment, "avg_logprob") else 0.9,
            ))

        if detected_language == "auto":
            detected_language = getattr(info, "language", None) or "unknown"

        return Transcript(language=detected_language or "unknown", entries=entries)

    @staticmethod
    def _assign_speaker(start: float, end: float, diarization: list[dict]) -> str:
        """Find the diarization speaker with the greatest overlap with [start, end]."""
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for seg in diarization:
            seg_end = seg["end"] if seg["end"] != float("inf") else end + 1.0
            overlap = min(end, seg_end) - max(start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]
        return best_speaker

    def _analyze_emotions(self, audio_path: Path, transcript: Transcript) -> Transcript:
        """Analyze emotion for each transcript entry using the correct audio segment."""
        try:
            from transformers import pipeline

            if self._emotion_model is None:
                device_str = get_device()
                if device_str == "mps":
                    device = torch.device("mps")
                elif device_str == "cuda":
                    device = 0
                else:
                    device = -1
                self._emotion_model = pipeline(
                    "audio-classification",
                    model="laica-labs/emotion2vec_plus_large",
                    device=device,
                )

            print("    [Emotion] Analyzing emotions ...")
            audio_array, sample_rate = sf.read(str(audio_path), dtype="float32")

            for entry in transcript.entries:
                try:
                    start_sample = int(entry.start * sample_rate)
                    end_sample = int(entry.end * sample_rate)
                    chunk = audio_array[start_sample:end_sample]
                    if len(chunk) < int(sample_rate * 0.1):
                        continue
                    result = self._emotion_model({"raw": chunk, "sampling_rate": int(sample_rate)})
                    if result:
                        entry.emotion = result[0]["label"]
                        entry.confidence = max(entry.confidence, result[0]["score"])
                except Exception:
                    pass

            return transcript

        except Exception as e:
            print(f"    [Emotion] Error: {e}, skipping emotion analysis")
            return transcript