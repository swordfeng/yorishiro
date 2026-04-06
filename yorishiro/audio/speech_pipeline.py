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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

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
    stt_cpu_threads: int = 0    # 0 = use all available cores
    stt_num_workers: int = 1    # parallel CTranslate2 replicas (num_workers in WhisperModel)
    language: str | None = None
    emotion_backend: str = "emotion2vec"
    emotion_model: str = "emotion2vec/emotion2vec_plus_base"
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
        force: bool = False,
    ) -> Transcript:
        """Run STT using cached vad.json + diarization.json. Writes transcript_raw.json."""
        vad_path = output_dir / "vad.json"
        diar_path = output_dir / "diarization.json"
        speech_segments = json.loads(vad_path.read_text(encoding="utf-8"))
        diarization = json.loads(diar_path.read_text(encoding="utf-8"))
        detected_language = language or self.config.language
        print(f"  [STT] Transcribing {audio_path.name} ...")
        transcript = self._run_transcription(audio_path, diarization, speech_segments, detected_language, output_dir=output_dir, force=force)
        out = output_dir / "transcript_raw.json"
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [STT] Done — {len(transcript.entries)} segment(s), language: {transcript.language}")
        return transcript

    def run_emotion(self, audio_path: Path, output_dir: Path) -> Transcript:
        """Run emotion analysis on transcript_raw.json. Writes transcript.json."""
        import gc
        raw_path = output_dir / "transcript_raw.json"
        transcript = Transcript(**json.loads(raw_path.read_text(encoding="utf-8")))
        print(f"  [Emotion] Analyzing {len(transcript.entries)} segment(s) ...")
        transcript = self._analyze_emotions(audio_path, transcript)
        self._emotion_model = None   # free before prosody
        gc.collect()
        print(f"  [Prosody] Analyzing {len(transcript.entries)} segment(s) ...")
        transcript = self._analyze_prosody(audio_path, transcript)
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

    @staticmethod
    def _vad_chunk_boundaries(
        total_duration: float,
        vad_segments: list[dict],
        chunk_seconds: float = _DIARIZATION_CHUNK_SECONDS,
    ) -> list[float]:
        """Chunk boundary times snapped to nearest VAD silence gap midpoint.

        All silence regions are considered (before first segment, between segments,
        after last segment). Always cuts in silence — never falls back to raw target.
        Deduplicates boundaries if multiple targets snap to the same gap.
        """
        num_chunks = max(1, math.ceil(total_duration / chunk_seconds))
        if num_chunks == 1:
            return [0.0, total_duration]

        # Collect midpoints of all silence regions
        gap_mids: list[float] = []
        if vad_segments:
            if vad_segments[0]["start"] > 0:
                gap_mids.append(vad_segments[0]["start"] / 2)
            for j in range(len(vad_segments) - 1):
                gap_mids.append((vad_segments[j]["end"] + vad_segments[j + 1]["start"]) / 2)
            if vad_segments[-1]["end"] < total_duration:
                gap_mids.append((vad_segments[-1]["end"] + total_duration) / 2)

        boundaries: list[float] = [0.0]
        for i in range(1, num_chunks):
            target = i * (total_duration / num_chunks)
            if gap_mids:
                best_mid = min(gap_mids, key=lambda m: abs(m - target))
                boundaries.append(best_mid)
            else:
                boundaries.append(target)
        boundaries.append(total_duration)

        # Deduplicate while preserving order (two targets may snap to the same gap)
        seen: set[float] = set()
        deduped: list[float] = []
        for b in boundaries:
            if b not in seen:
                deduped.append(b)
                seen.add(b)
        return deduped

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
        """Merge per-chunk speaker labels into global IDs via clustering."""
        all_embeddings: list[np.ndarray] = []
        metadata: list[tuple[int, str]] = []  # (chunk_idx, local_speaker_id)

        for chunk_idx, chunk in enumerate(chunks):
            emb = chunk["embeddings"]
            speakers_local = chunk["speakers_local"]
            if emb is not None:
                for i, local_id in enumerate(speakers_local):
                    all_embeddings.append(emb[i])
                    metadata.append((chunk_idx, local_id))

        if len(all_embeddings) == 0:
            all_turns: list[dict] = []
            for chunk_idx, chunk in enumerate(chunks):
                for turn in chunk["turns"]:
                    all_turns.append({**turn, "speaker": f"SPEAKER_{chunk_idx:02d}_{turn['speaker']}"})
            all_turns.sort(key=lambda t: t["start"])
            return all_turns

        X = np.stack(all_embeddings)
        distances = pdist(X, metric="cosine")
        Z = linkage(distances, method="average")
        threshold = 1.0 - _SPEAKER_SIM_THRESHOLD
        cluster_labels = fcluster(Z, t=threshold, criterion="distance")

        unique_clusters = np.unique(cluster_labels)
        cluster_to_speaker = {c: f"SPEAKER_{i:02d}" for i, c in enumerate(unique_clusters)}

        local_to_global: dict[tuple[int, str], str] = {}
        for (chunk_idx, local_id), cluster_label in zip(metadata, cluster_labels):
            local_to_global[(chunk_idx, local_id)] = cluster_to_speaker[cluster_label]

        print(f"    [Diarization] Clustered {len(all_embeddings)} local speakers into {len(unique_clusters)} global speakers")

        all_turns = []
        for chunk_idx, chunk in enumerate(chunks):
            for turn in chunk["turns"]:
                global_speaker = local_to_global.get((chunk_idx, turn["speaker"]), turn["speaker"])
                all_turns.append({**turn, "speaker": global_speaker})

        all_turns.sort(key=lambda t: t["start"])
        return all_turns

    def _run_diarization(self, audio_path: Path, output_dir: Path | None = None, force: bool = False) -> list[dict]:
        """Run speaker diarization with per-chunk checkpointing."""
        try:
            hf_token = os.environ.get(self.config.hf_token_env)
            if not hf_token:
                print(f"    [Diarization] {self.config.hf_token_env} not set, using single-speaker fallback")
                return [{"speaker": "SPEAKER_00", "start": 0.0, "end": float("inf")}]

            info = sf.info(str(audio_path))
            total_duration = info.duration
            sample_rate = info.samplerate
            source_mtime = audio_path.stat().st_mtime

            # Load VAD to find silence gaps for clean chunk boundaries
            vad_segments: list[dict] = []
            if output_dir and (output_dir / "vad.json").exists():
                vad_segments = json.loads((output_dir / "vad.json").read_text(encoding="utf-8"))

            # Build chunk boundaries snapped to VAD silence gaps
            boundaries = self._vad_chunk_boundaries(total_duration, vad_segments)
            num_chunks = len(boundaries) - 1

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
                chunk_audio, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")

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
        speech_segments: list[dict],
        language: str | None,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> Transcript:
        """Run speech-to-text in chunks to bound peak memory, then assign speakers by overlap."""
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
            cpu_threads = self.config.stt_cpu_threads or os.cpu_count() or 4
            num_workers = self.config.stt_num_workers
            self._whisper_model = WhisperModel(
                self.config.stt_model, device=device,
                cpu_threads=cpu_threads, num_workers=num_workers,
            )
            print(f"    [STT] Loaded {self.config.stt_model} on {device} ({cpu_threads} threads × {num_workers} workers)")

        import librosa

        with sf.SoundFile(str(audio_path)) as f:
            file_sample_rate = f.samplerate
            total_duration = f.frames / file_sample_rate

        boundaries = self._vad_chunk_boundaries(total_duration, speech_segments)

        # Merge any pure-silence chunk into its neighbour so no time is unaccounted for.
        # Pass left-to-right: if [prev, b] has no speech, drop b (silence absorbed into next).
        # Then pass right-to-left for trailing silence: if [b, next] has no speech, drop b.
        def _has_speech(start: float, end: float) -> bool:
            return any(s["start"] < end and s["end"] > start for s in speech_segments)

        merged = [boundaries[0]]
        for b in boundaries[1:-1]:
            if _has_speech(merged[-1], b):
                merged.append(b)
            # else: drop b — this chunk is silent, absorbed into the next
        merged.append(boundaries[-1])
        # Handle trailing silence: merge backward
        while len(merged) > 2 and not _has_speech(merged[-2], merged[-1]):
            merged.pop(-2)
        boundaries = merged
        num_chunks = len(boundaries) - 1

        source_mtime = audio_path.stat().st_mtime
        ckpt_dir = output_dir / ".stt_checkpoints" if output_dir else None

        if force and ckpt_dir and ckpt_dir.exists():
            import shutil
            shutil.rmtree(ckpt_dir)
            print("    [STT] Cleared checkpoints (force)")

        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        from concurrent.futures import ThreadPoolExecutor

        assert self._whisper_model is not None
        whisper_model = self._whisper_model

        def _transcribe_chunk(chunk_idx: int) -> tuple[int, list[TranscriptEntry], str | None]:
            chunk_start = boundaries[chunk_idx]
            chunk_end = boundaries[chunk_idx + 1]

            ckpt_file = ckpt_dir / f"chunk_{chunk_idx:04d}.json" if ckpt_dir else None
            if ckpt_file and ckpt_file.exists():
                ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
                if ckpt.get("source_mtime") == source_mtime:
                    print(f"    [STT] chunk {chunk_idx + 1}/{num_chunks} — resuming from checkpoint")
                    return chunk_idx, [TranscriptEntry(**e) for e in ckpt["entries"]], ckpt.get("detected_language")
            # Read only this chunk from disk — avoids loading the full file into RAM.
            # Each thread opens its own file handle to allow concurrent seeks.
            start_frame = int(chunk_start * file_sample_rate)
            end_frame = int(chunk_end * file_sample_rate)
            with sf.SoundFile(str(audio_path)) as fh:
                fh.seek(start_frame)
                chunk_audio = fh.read(end_frame - start_frame, dtype="float32")
            if chunk_audio.ndim > 1:
                chunk_audio = chunk_audio.mean(axis=1)
            if file_sample_rate != 16000:
                chunk_audio = librosa.resample(chunk_audio, orig_sr=file_sample_rate, target_sr=16000)
            print(f"    [STT] chunk {chunk_idx + 1}/{num_chunks}  {chunk_start:.0f}s–{chunk_end:.0f}s ...", flush=True)
            segments, info = whisper_model.transcribe(
                chunk_audio,
                language=language or None,
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            from tqdm import tqdm
            chunk_duration = chunk_end - chunk_start
            chunk_entries: list[TranscriptEntry] = []
            with tqdm(total=chunk_duration, desc=f"    [STT] chunk {chunk_idx + 1}/{num_chunks}", unit="s", bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
                last_end = 0.0
                for segment in segments:
                    abs_start = chunk_start + segment.start
                    abs_end = chunk_start + segment.end
                    chunk_entries.append(TranscriptEntry(
                        speaker_global=self._assign_speaker(abs_start, abs_end, diarization),
                        start=abs_start,
                        end=abs_end,
                        text=segment.text.strip(),
                        confidence=segment.avg_logprob if hasattr(segment, "avg_logprob") else 0.9,
                    ))
                    pbar.update(min(segment.end, chunk_duration) - last_end)
                    last_end = min(segment.end, chunk_duration)
            detected = getattr(info, "language", None)
            if ckpt_file:
                ckpt_data = {
                    "chunk_idx": chunk_idx,
                    "source_mtime": source_mtime,
                    "entries": [e.model_dump() for e in chunk_entries],
                    "detected_language": detected,
                }
                ckpt_file.write_text(json.dumps(ckpt_data, ensure_ascii=False, indent=2), encoding="utf-8")
            return chunk_idx, chunk_entries, detected

        num_workers = self.config.stt_num_workers
        chunk_results: list[tuple[int, list[TranscriptEntry], str | None]] = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_transcribe_chunk, i) for i in range(num_chunks)]
            for f in futures:
                chunk_results.append(f.result())

        chunk_results.sort(key=lambda t: t[0])
        entries: list[TranscriptEntry] = []
        detected_language: str | None = language
        for _, chunk_entries, lang in chunk_results:
            entries.extend(chunk_entries)
            if detected_language is None and lang:
                detected_language = lang

        return Transcript(
            language=detected_language or "unknown",
            entries=entries,
        )

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
        """Analyze emotion for each transcript entry. Reads segments lazily, caps at 10s each."""
        import contextlib
        import gc
        import io
        import time
        from funasr import AutoModel
        from tqdm import tqdm

        try:
            if self._emotion_model is None:
                self._emotion_model = AutoModel(
                    model=self.config.emotion_model,
                    hub="hf",
                    disable_update=True,
                    device=get_device(),
                )
        except Exception as e:
            print(f"    [Emotion] Failed to load model: {e}")
            return transcript

        info = sf.info(str(audio_path))
        sample_rate = info.samplerate
        max_samples = sample_rate * 10

        total = len(transcript.entries)
        errors = 0
        total_inference_time = 0.0

        print(f"    [Emotion] Analyzing {total} segment(s) ...")
        with tqdm(total=total, desc="    [Emotion]", unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i, entry in enumerate(transcript.entries):
                seg_duration = entry.end - entry.start
                pbar.set_postfix_str(f"seg={seg_duration:.1f}s")

                start_sample = int(entry.start * sample_rate)
                end_sample = min(int(entry.end * sample_rate), start_sample + max_samples)
                if end_sample - start_sample < int(sample_rate * 0.1):
                    pbar.update(1)
                    continue

                try:
                    chunk, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")
                    infer_start = time.perf_counter()
                    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = self._emotion_model.generate(
                            input=chunk,
                            sample_rate=int(sample_rate),
                            granularity="utterance",
                            extract_embedding=False,
                        )
                    infer_time = time.perf_counter() - infer_start
                    total_inference_time += infer_time

                    if result and result[0].get("scores"):
                        scores = result[0]["scores"]
                        labels = result[0]["labels"]
                        best_idx = int(max(range(len(scores)), key=lambda j: scores[j]))
                        raw_label = labels[best_idx]
                        entry.emotion = raw_label.split("/")[-1] if "/" in raw_label else raw_label
                        entry.confidence = max(entry.confidence, scores[best_idx])
                except Exception as e:
                    errors += 1
                    pbar.write(f"    [Emotion] Warning: entry {i} failed: {e}")

                if i % 20 == 0:
                    gc.collect()
                pbar.update(1)

        print(f"    [Emotion] Done — {total} segment(s), {errors} error(s), {total_inference_time:.1f}s inference")
        return transcript

    def _analyze_prosody(self, audio_path: Path, transcript: Transcript) -> Transcript:
        """Classify pitch trend, speech rate, and volume for each transcript entry.

        Pure librosa signal processing — no ML model, no GPU.
        Same lazy per-entry sf.read(start=, stop=) pattern as _analyze_emotions.
        """
        import gc
        import librosa
        from tqdm import tqdm

        info = sf.info(str(audio_path))
        sr = info.samplerate
        max_samples = sr * 10

        total = len(transcript.entries)
        errors = 0

        print(f"    [Prosody] Analyzing {total} segment(s) ...")
        with tqdm(total=total, desc="    [Prosody]", unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i, entry in enumerate(transcript.entries):
                seg_duration = entry.end - entry.start
                pbar.set_postfix_str(f"seg={seg_duration:.1f}s")

                start_sample = int(entry.start * sr)
                end_sample = min(int(entry.end * sr), start_sample + max_samples)
                if end_sample - start_sample < int(sr * 0.1):
                    pbar.update(1)
                    continue

                try:
                    chunk, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")
                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)

                    # volume — mean RMS in dBFS
                    rms_frames = librosa.feature.rms(y=chunk)[0]
                    db = 20.0 * np.log10(float(np.mean(rms_frames)) + 1e-9)
                    entry.volume = "quiet" if db < -38.0 else ("loud" if db > -20.0 else "normal")

                    # speech_rate — onset density per second
                    duration = entry.end - entry.start
                    if duration >= 0.3:
                        onsets = librosa.onset.onset_detect(y=chunk, sr=sr, units="time", normalize=True)
                        rate = len(onsets) / duration
                        entry.speech_rate = "slow" if rate < 2.0 else ("fast" if rate > 4.0 else "normal")

                    # pitch_trend — pyin F0, classify by slope + CoV
                    f0, voiced_flag, _ = librosa.pyin(
                        chunk,
                        fmin=float(librosa.note_to_hz('C2')),
                        fmax=float(librosa.note_to_hz('C7')),
                        sr=sr,
                    )
                    voiced_f0 = f0[voiced_flag]
                    if len(voiced_f0) >= 4:
                        mean_f0 = float(np.mean(voiced_f0))
                        rel_std = float(np.std(voiced_f0)) / mean_f0
                        norm_slope = float(np.polyfit(np.arange(len(voiced_f0)), voiced_f0, 1)[0]) / mean_f0
                        if rel_std > 0.25:
                            entry.pitch_trend = "variable"
                        elif norm_slope > 0.003:
                            entry.pitch_trend = "rising"
                        elif norm_slope < -0.003:
                            entry.pitch_trend = "falling"
                        else:
                            entry.pitch_trend = "steady"

                except Exception as e:
                    errors += 1
                    pbar.write(f"    [Prosody] Warning: entry {i} failed: {e}")

                if i % 20 == 0:
                    gc.collect()
                pbar.update(1)

        print(f"    [Prosody] Done — {total} segment(s), {errors} error(s)")
        return transcript