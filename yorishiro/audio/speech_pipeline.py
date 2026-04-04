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
from dataclasses import dataclass

from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.utils import get_device


@dataclass
class SpeechPipelineConfig:
    vad_backend: str = "silero-vad"
    diarization_backend: str = "pyannote"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    stt_backend: str = "faster-whisper"
    stt_model: str = "large-v3"
    language: str | None = None
    emotion_backend: str = "emotion2vec"
    hf_token_env: str = "YORISHIRO_HF_TOKEN"


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
        audio_path: Path | None = None,
    ) -> Transcript:
        """Process audio from video and return transcript.

        If audio_path is provided (e.g. a pre-separated voice.flac), it is used
        directly and audio extraction from video is skipped.
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

        if audio_path is not None:
            print(f"  [SpeechPipeline] Using pre-extracted audio: {audio_path.name}")
        else:
            print(f"  [SpeechPipeline] Extracting audio from {video_path.name} ...")
            audio_path = self._extract_audio(video_path, output_dir)
            print(f"  [SpeechPipeline] Audio extracted: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

        print("  [SpeechPipeline] Running VAD ...")
        speech_segments = self._run_vad(audio_path)
        print(f"  [SpeechPipeline] VAD: {len(speech_segments)} speech segment(s)")

        print("  [SpeechPipeline] Running speaker diarization ...")
        diarization = self._run_diarization(audio_path)
        speakers = {d["speaker"] for d in diarization}
        print(f"  [SpeechPipeline] Diarization: {len(diarization)} turn(s), {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")

        print("  [SpeechPipeline] Running transcription ...")
        detected_language = language or self.config.language
        transcript = self._run_transcription(audio_path, diarization, speech_segments, detected_language)
        print(f"  [SpeechPipeline] Transcription: {len(transcript.entries)} segment(s), language: {transcript.language}")

        print("  [SpeechPipeline] Running emotion analysis ...")
        transcript = self._analyze_emotions(audio_path, transcript)
        emotions = {e.emotion for e in transcript.entries if e.emotion}
        print(f"  [SpeechPipeline] Emotions detected: {', '.join(sorted(emotions)) if emotions else 'none'}")

        cache_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [SpeechPipeline] Done — {len(transcript.entries)} segments cached")
        return transcript

    # ------------------------------------------------------------------
    # Public single-stage methods (used by individual step classes)
    # ------------------------------------------------------------------

    def run_vad(self, audio_path: Path, output_dir: Path) -> list[dict]:
        """Run VAD and write vad.json. Returns speech segments."""
        print(f"  [VAD] Running on {audio_path.name} ...")
        segments = self._run_vad(audio_path)
        out = output_dir / "vad.json"
        out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [VAD] Done — {len(segments)} speech segment(s)")
        return segments

    def run_diarization(self, audio_path: Path, output_dir: Path) -> list[dict]:
        """Run diarization and write diarization.json. Returns speaker turns."""
        print(f"  [Diarization] Running on {audio_path.name} ...")
        turns = self._run_diarization(audio_path)
        out = output_dir / "diarization.json"
        out.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        speakers = {t["speaker"] for t in turns}
        print(f"  [Diarization] Done — {len(turns)} turn(s), {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")
        return turns

    def run_stt(
        self,
        audio_path: Path,
        output_dir: Path,
        language: str | None = None,
    ) -> Transcript:
        """Run STT using cached vad.json + diarization.json. Writes transcript_raw.json."""
        vad_path = output_dir / "vad.json"
        diar_path = output_dir / "diarization.json"
        speech_segments = json.loads(vad_path.read_text(encoding="utf-8"))
        diarization = json.loads(diar_path.read_text(encoding="utf-8"))
        detected_language = language or self.config.language
        print(f"  [STT] Transcribing {audio_path.name} ...")
        transcript = self._run_transcription(audio_path, diarization, speech_segments, detected_language)
        out = output_dir / "transcript_raw.json"
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [STT] Done — {len(transcript.entries)} segment(s), language: {transcript.language}")
        return transcript

    def run_emotion(self, audio_path: Path, output_dir: Path) -> Transcript:
        """Run emotion analysis on transcript_raw.json. Writes transcript.json."""
        raw_path = output_dir / "transcript_raw.json"
        transcript = Transcript(**json.loads(raw_path.read_text(encoding="utf-8")))
        print(f"  [Emotion] Analyzing {len(transcript.entries)} segment(s) ...")
        transcript = self._analyze_emotions(audio_path, transcript)
        emotions = {e.emotion for e in transcript.entries if e.emotion}
        out = output_dir / "transcript.json"
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [Emotion] Done — {', '.join(sorted(emotions)) if emotions else 'none'}")
        return transcript

    def _extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Extract audio from video as FLAC using PyAV."""
        audio_path = output_dir / "audio.flac"

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
        output_stream = output_container.add_stream("flac", rate=16000)
        assert isinstance(output_stream, av.AudioStream)
        output_stream.layout = "mono"

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in input_container.decode(audio_stream):
            assert isinstance(frame, AudioFrame)
            for resampled in resampler.resample(frame):
                for packet in output_stream.encode(resampled):
                    output_container.mux(packet)
        for resampled in resampler.resample(None):
            for packet in output_stream.encode(resampled):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()

        return audio_path

    def _run_vad(self, audio_path: Path) -> list[dict]:
        """Run Voice Activity Detection."""
        from silero_vad import load_silero_vad, read_audio
        from silero_vad import get_speech_timestamps

        model = load_silero_vad()
        wav = read_audio(str(audio_path))

        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
        speech_segments = [{"start": t["start"], "end": t["end"]} for t in timestamps]

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
            from tqdm import tqdm
            with tqdm(total=1.0, desc="    [Diarization]", unit="%", bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
                last = 0.0
                def _hook(_step_name: str, _step_artifact: object, file: object = None, total: int | None = None, completed: int | None = None) -> None:  # noqa: ARG001
                    nonlocal last
                    if completed is not None and total is not None and total > 0:
                        progress = completed / total
                        pbar.update(progress - last)
                        last = progress
                diarization = self._diarization_pipeline(str(audio_path), hook=_hook)

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
        speech_segments: list[dict],  # noqa: ARG002 — reserved for future VAD-gated transcription
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