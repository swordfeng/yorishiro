"""Speech transcription, speaker diarization, and emotion analysis.

Pipeline:
1. VAD preprocessing (Silero-VAD)
2. Speaker diarization (pyannote-audio)
3. Speech-to-text (faster-whisper)
4. Emotion/tonality analysis (emotion2vec)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import torch
from av.audio.frame import AudioFrame
from dataclasses import dataclass

from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.utils import get_device

_DIARIZATION_CHUNK_SECONDS = 1200.0   # target chunk duration; actual cuts snap to VAD silence gaps
_SPEAKER_SIM_THRESHOLD = 0.75         # cosine similarity threshold for cross-chunk speaker matching


@dataclass
class SpeechPipelineConfig:
    vad_backend: str = "silero-vad"
    diarization_backend: str = "pyannote"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_batch_size: int = 32    # pyannote 4.x default is 1; 32 is much faster
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

    def run_diarization(self, audio_path: Path, output_dir: Path, force: bool = False) -> list[dict]:
        """Run diarization and write diarization.json. Returns speaker turns."""
        print(f"  [Diarization] Running on {audio_path.name} ...")
        turns = self._run_diarization(audio_path, output_dir=output_dir, force=force)
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

    def _load_diarization_pipeline(self, hf_token: str) -> object:
        """Load and cache the pyannote pipeline with batch size settings."""
        from pyannote.audio import Pipeline

        if self._diarization_pipeline is None:
            pipeline = Pipeline.from_pretrained(self.config.diarization_model, token=hf_token)
            device = torch.device(get_device())
            pipeline = pipeline.to(device)  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
            if hasattr(pipeline, '_segmentation'):
                pipeline._segmentation.batch_size = self.config.diarization_batch_size
            if hasattr(pipeline, 'embedding_batch_size'):
                pipeline.embedding_batch_size = self.config.diarization_batch_size
            self._diarization_pipeline = pipeline
        return self._diarization_pipeline

    def _diarize_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_idx: int,
        chunk_start: float,  # absolute time of audio[0]; added to segment times
    ) -> tuple[list[dict], np.ndarray | None, list[str]]:
        """Diarize one audio chunk. chunk_start is the absolute offset of audio[0]."""
        from tqdm import tqdm
        import tempfile

        assert self._diarization_pipeline is not None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            sf.write(str(tmp_path), audio, sample_rate)
            with tqdm(total=1.0, desc=f"    [Diarization] chunk {chunk_idx}", unit="%", bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
                last = 0.0
                def _hook(_step_name: str, _step_artifact: object, file: object = None, total: int | None = None, completed: int | None = None) -> None:  # noqa: ARG001
                    nonlocal last
                    if completed is not None and total is not None and total > 0:
                        progress = completed / total
                        pbar.update(progress - last)
                        last = progress
                result = self._diarization_pipeline(str(tmp_path), hook=_hook)

            ann = result.exclusive_speaker_diarization if hasattr(result, 'exclusive_speaker_diarization') else result
            embeddings: np.ndarray | None = result.speaker_embeddings if hasattr(result, 'speaker_embeddings') else None
            # speaker_embeddings rows are ordered by speaker_diarization.labels(), not exclusive_speaker_diarization.labels()
            full_ann = result.speaker_diarization if hasattr(result, 'speaker_diarization') else ann
            speakers_local = full_ann.labels() if hasattr(full_ann, 'labels') else []

            turns = []
            for segment, _, speaker in ann.itertracks(yield_label=True):
                turns.append({
                    "speaker": speaker,
                    "start": round(chunk_start + segment.start, 3),
                    "end": round(chunk_start + segment.end, 3),
                })
            return turns, embeddings, speakers_local
        finally:
            tmp_path.unlink(missing_ok=True)

    def _merge_chunk_speakers(
        self,
        chunks: list[dict],  # each: {turns, embeddings, speakers_local}
    ) -> list[dict]:
        """Merge per-chunk speaker labels into global IDs via embedding cosine similarity."""
        global_embeddings: list[np.ndarray] = []   # one per global speaker
        global_labels: list[str] = []              # e.g. "SPEAKER_00", "SPEAKER_01", ...
        local_to_global: list[dict[str, str]] = []  # per chunk

        for chunk in chunks:
            embeddings: np.ndarray | None = chunk["embeddings"]
            speakers_local: list[str] = chunk["speakers_local"]
            mapping: dict[str, str] = {}

            for i, local_id in enumerate(speakers_local):
                emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None

                best_global: str | None = None
                best_sim = -1.0
                if emb is not None:
                    norm = np.linalg.norm(emb)
                    emb_n = emb / norm if norm > 0 else emb
                    for j, g_emb in enumerate(global_embeddings):
                        g_norm = np.linalg.norm(g_emb)
                        g_emb_n = g_emb / g_norm if g_norm > 0 else g_emb
                        sim = float(np.dot(emb_n, g_emb_n))
                        if sim > best_sim:
                            best_sim = sim
                            best_global = global_labels[j]

                if best_global is not None and best_sim >= _SPEAKER_SIM_THRESHOLD:
                    mapping[local_id] = best_global
                    # Update running average of the matched global embedding
                    idx = global_labels.index(best_global)
                    global_embeddings[idx] = (global_embeddings[idx] + emb) / 2
                else:
                    new_label = f"SPEAKER_{len(global_labels):02d}"
                    mapping[local_id] = new_label
                    global_labels.append(new_label)
                    global_embeddings.append(emb if emb is not None else np.zeros(1))

            local_to_global.append(mapping)

        # Apply mappings and flatten turns
        all_turns: list[dict] = []
        for chunk, mapping in zip(chunks, local_to_global):
            for turn in chunk["turns"]:
                all_turns.append({**turn, "speaker": mapping.get(turn["speaker"], turn["speaker"])})
        all_turns.sort(key=lambda t: t["start"])
        return all_turns

    def _run_diarization(self, audio_path: Path, output_dir: Path | None = None, force: bool = False) -> list[dict]:
        """Run speaker diarization with per-chunk checkpointing."""
        try:
            hf_token = os.environ.get(self.config.hf_token_env)
            if not hf_token:
                print(f"    [Diarization] {self.config.hf_token_env} not set, using single-speaker fallback")
                return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

            audio, sample_rate = sf.read(str(audio_path), dtype="float32")
            total_duration = len(audio) / sample_rate
            source_mtime = audio_path.stat().st_mtime

            # Load VAD to find silence gaps for clean chunk boundaries
            vad_segments: list[dict] = []
            if output_dir and (output_dir / "vad.json").exists():
                vad_segments = json.loads((output_dir / "vad.json").read_text(encoding="utf-8"))

            # Build chunk boundaries snapped to VAD silence gaps
            # A silence gap is the interval between vad_segments[i]["end"] and vad_segments[i+1]["start"]
            # For each target boundary, pick the nearest silence gap midpoint
            num_chunks = max(1, math.ceil(total_duration / _DIARIZATION_CHUNK_SECONDS))
            boundaries = [0.0]
            for i in range(1, num_chunks):
                target = i * (total_duration / num_chunks)
                if len(vad_segments) >= 2:
                    best_mid = target
                    for j in range(len(vad_segments) - 1):
                        gap_mid = (vad_segments[j]["end"] + vad_segments[j + 1]["start"]) / 2
                        if abs(gap_mid - target) < abs(best_mid - target):
                            best_mid = gap_mid
                    boundaries.append(best_mid)
                else:
                    boundaries.append(target)
            boundaries.append(total_duration)

            # Checkpoint directory
            ckpt_dir = output_dir / ".diarization_checkpoints" if output_dir else None

            if force:
                import shutil
                if ckpt_dir and ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                    print("    [Diarization] Cleared checkpoints (force)")
                if output_dir:
                    out_json = output_dir / "diarization.json"
                    if out_json.exists():
                        out_json.unlink()
                        print("    [Diarization] Cleared output (force)")

            if ckpt_dir:
                ckpt_dir.mkdir(parents=True, exist_ok=True)

            self._load_diarization_pipeline(hf_token)

            chunk_results: list[dict] = []
            for chunk_idx in range(num_chunks):
                chunk_start = boundaries[chunk_idx]
                chunk_end = boundaries[chunk_idx + 1]

                ckpt_file = ckpt_dir / f"chunk_{chunk_idx:04d}.json" if ckpt_dir else None

                # Check checkpoint validity
                if ckpt_file and ckpt_file.exists() and not force:
                    ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
                    if ckpt.get("source_mtime") == source_mtime:
                        print(f"    [Diarization] chunk {chunk_idx} — resuming from checkpoint")
                        # Load embeddings: prefer .npy, fall back to legacy JSON field
                        npy_file = ckpt_file.with_suffix(".npy")
                        if npy_file.exists():
                            ckpt["embeddings"] = np.load(str(npy_file))
                        elif ckpt.get("embeddings") is not None:
                            # Migrate: save as .npy and remove from JSON
                            emb = np.array(ckpt["embeddings"], dtype=np.float32)
                            np.save(str(npy_file), emb)
                            ckpt["embeddings"] = emb
                            del ckpt["embeddings"]  # will be reloaded from npy next time
                            ckpt_no_emb = {k: v for k, v in ckpt.items() if k != "embeddings"}
                            ckpt_file.write_text(json.dumps(ckpt_no_emb, ensure_ascii=False, indent=2), encoding="utf-8")
                            ckpt["embeddings"] = emb
                        else:
                            ckpt["embeddings"] = None
                        chunk_results.append(ckpt)
                        continue
                    else:
                        print(f"    [Diarization] chunk {chunk_idx} — checkpoint stale, reprocessing")

                start_sample = int(chunk_start * sample_rate)
                end_sample = int(chunk_end * sample_rate)
                chunk_audio = audio[start_sample:end_sample]

                turns, embeddings, speakers_local = self._diarize_chunk(
                    chunk_audio, sample_rate, chunk_idx, chunk_start,
                )

                ckpt_data: dict = {
                    "chunk_idx": chunk_idx,
                    "source_mtime": source_mtime,
                    "turns": turns,
                    "speakers_local": speakers_local,
                    "embeddings": embeddings,
                }
                if ckpt_file:
                    ckpt_json = {k: v for k, v in ckpt_data.items() if k != "embeddings"}
                    ckpt_file.write_text(json.dumps(ckpt_json, ensure_ascii=False, indent=2), encoding="utf-8")
                    if embeddings is not None:
                        np.save(str(ckpt_file.with_suffix(".npy")), embeddings)

                chunk_results.append(ckpt_data)

            if num_chunks == 1:
                # No merging needed — remap to canonical names
                all_turns = []
                mapping = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(sorted(set(t["speaker"] for t in chunk_results[0]["turns"])))}
                for t in chunk_results[0]["turns"]:
                    all_turns.append({**t, "speaker": mapping.get(t["speaker"], t["speaker"])})
                segments = all_turns
            else:
                segments = self._merge_chunk_speakers(chunk_results)

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