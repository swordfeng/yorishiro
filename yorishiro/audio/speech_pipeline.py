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
import re
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
_STT_WORD_GAP_SPLIT_SECONDS = 0.35
_STT_TEXT_SPLIT_MIN_CHARS = 24
_STT_JA_CLAUSE_ENDINGS = (
    "けれども",
    "けども",
    "けれど",
    "だったり",
    "でしたり",
    "なくて",
    "たり",
    "だり",
    "けど",
    "ので",
    "のに",
    "から",
    "して",
    "くて",
    "て",
    "で",
)
_STT_PUNCT_SPLIT_RE = re.compile(r"(?<=[。！？!?、,])")


def _prosody_segment_worker(args):
    """Worker for multiprocessing prosody analysis."""
    idx_info, audio_path_str, sr, max_samples = args
    
    i = idx_info["i"]
    start_sample = idx_info["start_sample"]
    end_sample = idx_info["end_sample"]
    
    if end_sample - start_sample < int(sr * 0.1):
        return (i, None, None, None, None, None)
    
    import numpy as np
    import librosa
    
    try:
        chunk, _ = sf.read(audio_path_str, start=start_sample, stop=end_sample, dtype="float32")
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        
        volume = None
        speech_rate = None
        pitch_trend = None
        
        # Volume
        rms_frames = librosa.feature.rms(y=chunk)[0]
        db = 20.0 * np.log10(float(np.mean(rms_frames)) + 1e-9)
        volume = "quiet" if db < -38.0 else ("loud" if db > -20.0 else "normal")
        
        # Speech rate
        duration = idx_info["duration"]
        if duration >= 0.3:
            onsets = librosa.onset.onset_detect(y=chunk, sr=sr, units="time", normalize=True)
            rate = len(onsets) / duration
            speech_rate = "slow" if rate < 2.0 else ("fast" if rate > 4.0 else "normal")
        
        # Pitch trend
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
                pitch_trend = "variable"
            elif norm_slope > 0.003:
                pitch_trend = "rising"
            elif norm_slope < -0.003:
                pitch_trend = "falling"
            else:
                pitch_trend = "steady"
        
        return (i, volume, speech_rate, pitch_trend, None, None)
    
    except Exception as e:
        return (i, None, None, None, None, str(e))


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
    stt_word_timestamps: bool = False
    stt_vad_filter: bool = False
    stt_vad_min_silence_duration_ms: int = 500
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
    ) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Merge per-chunk speaker labels into global IDs via clustering.

        Returns (all_turns, speaker_embeddings) where speaker_embeddings maps
        global speaker ID → mean embedding vector.
        """
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
            return all_turns, {}

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

        # Compute mean embedding per global speaker
        speaker_emb_accum: dict[str, list[np.ndarray]] = {}
        for emb, cluster_label in zip(all_embeddings, cluster_labels):
            global_id = cluster_to_speaker[cluster_label]
            speaker_emb_accum.setdefault(global_id, []).append(emb)
        speaker_embeddings = {
            spk: np.mean(np.stack(embs), axis=0)
            for spk, embs in speaker_emb_accum.items()
        }

        print(f"    [Diarization] Clustered {len(all_embeddings)} local speakers into {len(unique_clusters)} global speakers")

        all_turns = []
        for chunk_idx, chunk in enumerate(chunks):
            for turn in chunk["turns"]:
                global_speaker = local_to_global.get((chunk_idx, turn["speaker"]), turn["speaker"])
                all_turns.append({**turn, "speaker": global_speaker})

        all_turns.sort(key=lambda t: t["start"])
        return all_turns, speaker_embeddings

    @staticmethod
    def _save_speaker_bank(
        output_dir: Path,
        segments: list[dict],
        speaker_embeddings: dict[str, np.ndarray],
    ) -> None:
        """Save a SpeakerBankManager populated with diarization results."""
        from yorishiro.audio.speaker_bank import SpeakerBankManager

        bank = SpeakerBankManager()
        speakers = sorted({t["speaker"] for t in segments})
        first_appearance = {}
        for t in segments:
            if t["speaker"] not in first_appearance:
                first_appearance[t["speaker"]] = t["start"]

        for spk_id in speakers:
            bank.speaker_bank.add_speaker(spk_id, first_appearance.get(spk_id, 0.0))
            if spk_id in speaker_embeddings:
                bank._embeddings[spk_id] = speaker_embeddings[spk_id]

        bank.save(output_dir)
        print(f"    [Diarization] Speaker bank saved with {len(speakers)} speaker(s)")

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
                sorted_speakers = sorted(set(t["speaker"] for t in chunk_results[0]["turns"]))
                mapping = {s: f"SPEAKER_{i:02d}" for i, s in enumerate(sorted_speakers)}
                for t in chunk_results[0]["turns"]:
                    all_turns.append({**t, "speaker": mapping.get(t["speaker"], t["speaker"])})
                segments = all_turns

                # Build per-speaker embeddings from single chunk
                speaker_embeddings: dict[str, np.ndarray] = {}
                emb = chunk_results[0].get("embeddings")
                speakers_local = chunk_results[0].get("speakers_local", [])
                if emb is not None:
                    for i, local_id in enumerate(speakers_local):
                        global_id = mapping.get(local_id, local_id)
                        speaker_embeddings[global_id] = emb[i]
            else:
                segments, speaker_embeddings = self._merge_chunk_speakers(chunk_results)

            # Populate speaker bank with discovered speakers and embeddings
            if output_dir and segments:
                self._save_speaker_bank(output_dir, segments, speaker_embeddings)

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
            transcribe_kwargs = {
                "language": language or None,
                "task": "transcribe",
                "vad_filter": self.config.stt_vad_filter,
                "word_timestamps": self.config.stt_word_timestamps,
                "condition_on_previous_text": False,
            }
            if self.config.stt_vad_filter:
                transcribe_kwargs["vad_parameters"] = {
                    "min_silence_duration_ms": self.config.stt_vad_min_silence_duration_ms,
                }

            try:
                segments, info = whisper_model.transcribe(chunk_audio, **transcribe_kwargs)
            except MemoryError:
                if not self.config.stt_word_timestamps:
                    raise
                print(
                    f"    [STT] chunk {chunk_idx + 1}/{num_chunks} — word timestamp alignment ran out of memory; retrying without word timestamps",
                    flush=True,
                )
                transcribe_kwargs["word_timestamps"] = False
                segments, info = whisper_model.transcribe(chunk_audio, **transcribe_kwargs)
            from tqdm import tqdm
            chunk_duration = chunk_end - chunk_start
            chunk_entries: list[TranscriptEntry] = []
            with tqdm(total=chunk_duration, desc=f"    [STT] chunk {chunk_idx + 1}/{num_chunks}", unit="s", bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
                last_end = 0.0
                detected = getattr(info, "language", None)
                chunk_language = language or detected
                for segment in segments:
                    for entry in self._segment_to_entries(segment, chunk_start, diarization, chunk_language):
                        chunk_entries.append(entry)
                    pbar.update(min(segment.end, chunk_duration) - last_end)
                    last_end = min(segment.end, chunk_duration)
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

    def _segment_to_entries(
        self,
        segment,
        chunk_start: float,
        diarization: list[dict],
        language: str | None,
    ) -> list[TranscriptEntry]:
        text = segment.text.strip()
        if not text:
            return []

        confidence = segment.avg_logprob if hasattr(segment, "avg_logprob") else 0.9
        word_entries = self._split_segment_from_words(segment, chunk_start, diarization, confidence, language)
        if word_entries:
            return word_entries

        abs_start = chunk_start + segment.start
        abs_end = chunk_start + segment.end
        return [
            TranscriptEntry(
                speaker_global=self._assign_speaker(abs_start, abs_end, diarization),
                start=abs_start,
                end=abs_end,
                text=text,
                confidence=confidence,
            )
        ]

    def _split_segment_from_words(
        self,
        segment,
        chunk_start: float,
        diarization: list[dict],
        confidence: float,
        language: str | None,
    ) -> list[TranscriptEntry]:
        words = getattr(segment, "words", None) or []
        timed_words: list[tuple[float, float, str]] = []
        for word in words:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            token = getattr(word, "word", "")
            if start is None or end is None:
                continue
            token = token.strip()
            if not token:
                continue
            timed_words.append((float(start), float(end), token))

        if len(timed_words) < 2:
            return []

        groups: list[list[tuple[float, float, str]]] = []
        current = [timed_words[0]]
        for prev, cur in zip(timed_words, timed_words[1:]):
            prev_end = prev[1]
            cur_start, cur_end, cur_token = cur
            pause = cur_start - prev_end
            should_split = pause >= _STT_WORD_GAP_SPLIT_SECONDS
            if not should_split:
                prev_token = prev[2]
                should_split = prev_token.endswith(("。", "！", "？", "!", "?", "、", ","))

            if should_split:
                groups.append(current)
                current = [cur]
            else:
                current.append(cur)
        groups.append(current)

        if len(groups) == 1:
            return []

        entries: list[TranscriptEntry] = []
        for group in groups:
            group_text = "".join(token for _, _, token in group).strip()
            if not group_text:
                continue
            abs_start = chunk_start + group[0][0]
            abs_end = chunk_start + group[-1][1]
            entries.extend(self._split_entry_text(abs_start, abs_end, group_text, confidence, diarization, language))
        return entries

    def _split_entry_text(
        self,
        start: float,
        end: float,
        text: str,
        confidence: float,
        diarization: list[dict],
        language: str | None,
    ) -> list[TranscriptEntry]:
        pieces = self._split_text_heuristically(text, language)
        if len(pieces) <= 1:
            return [
                TranscriptEntry(
                    speaker_global=self._assign_speaker(start, end, diarization),
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                )
            ]

        total_chars = sum(len(piece) for piece in pieces)
        duration = max(end - start, 0.0)
        cursor = start
        entries: list[TranscriptEntry] = []
        for idx, piece in enumerate(pieces):
            piece_duration = duration * (len(piece) / total_chars) if total_chars else 0.0
            piece_end = end if idx == len(pieces) - 1 else min(end, cursor + piece_duration)
            entries.append(TranscriptEntry(
                speaker_global=self._assign_speaker(cursor, piece_end, diarization),
                start=cursor,
                end=piece_end,
                text=piece,
                confidence=confidence,
            ))
            cursor = piece_end
        return entries

    @staticmethod
    def _split_text_heuristically(text: str, language: str | None) -> list[str]:
        pieces = [part.strip() for part in _STT_PUNCT_SPLIT_RE.split(text) if part.strip()]
        if len(pieces) > 1:
            return pieces
        if len(text) < _STT_TEXT_SPLIT_MIN_CHARS:
            return [text]
        normalized_language = SpeechPipeline._normalize_language(language)
        if normalized_language == "ja":
            return SpeechPipeline._split_japanese_clause_text(text)
        return [text]

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        if not language:
            return None
        normalized = language.strip().lower().replace("_", "-")
        if normalized.startswith("ja"):
            return "ja"
        if normalized.startswith("en"):
            return "en"
        if normalized.startswith("zh"):
            return "zh"
        if normalized.startswith("ko"):
            return "ko"
        return normalized

    @staticmethod
    def _split_japanese_clause_text(text: str) -> list[str]:
        break_points: list[int] = []
        min_piece_chars = 6
        for ending in _STT_JA_CLAUSE_ENDINGS:
            search_from = min_piece_chars
            while True:
                idx = text.find(ending, search_from)
                if idx < 0:
                    break
                split_at = idx + len(ending)
                left_len = split_at
                right_len = len(text) - split_at
                if left_len >= min_piece_chars and right_len >= min_piece_chars:
                    next_char = text[split_at]
                    if re.match(r"[一-龯ぁ-んァ-ヶー]", next_char):
                        break_points.append(split_at)
                search_from = split_at

        if not break_points:
            return [text]

        pieces: list[str] = []
        start = 0
        for split_at in sorted(set(break_points)):
            piece = text[start:split_at].strip()
            if piece:
                pieces.append(piece)
            start = split_at
        tail = text[start:].strip()
        if tail:
            pieces.append(tail)
        return pieces or [text]

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
        """Analyze emotion for each transcript entry using emotion2vec via funasr AutoModel."""
        import contextlib
        import gc
        import io
        import time
        from funasr import AutoModel
        from tqdm import tqdm

        info = sf.info(str(audio_path))
        file_sr = info.samplerate
        target_sr = 16000
        max_samples = int(file_sr * 10)

        total = len(transcript.entries)
        errors = 0
        total_inference_time = 0.0

        device_str = get_device()
        use_mps = torch.backends.mps.is_available()
        print(f"    [Emotion] Loading model on {device_str} ...")
        model = AutoModel(
            model=self.config.emotion_model,
            hub="hf",
            disable_update=True,
            device=device_str,
        )

        need_resample = file_sr != target_sr
        if need_resample:
            import librosa

        print(f"    [Emotion] Analyzing {total} segment(s) (file_sr={file_sr}) ...")
        with tqdm(total=total, desc="    [Emotion]", unit="seg",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for i, entry in enumerate(transcript.entries):
                seg_duration = entry.end - entry.start
                pbar.set_postfix_str(f"seg={seg_duration:.1f}s")

                start_sample = int(entry.start * file_sr)
                end_sample = min(int(entry.end * file_sr), start_sample + max_samples)
                if end_sample - start_sample < int(file_sr * 0.1):
                    pbar.update(1)
                    continue

                try:
                    chunk, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")
                except Exception as e:
                    errors += 1
                    pbar.write(f"    [Emotion] Warning: entry {i} read error: {e}")
                    pbar.update(1)
                    continue

                try:
                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)
                    if need_resample:
                        chunk = librosa.resample(chunk, orig_sr=file_sr, target_sr=target_sr)

                    infer_start = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result = model.generate(
                            input=chunk,
                            sample_rate=int(target_sr),
                            granularity="utterance",
                            extract_embedding=False,
                        )
                    total_inference_time += time.perf_counter() - infer_start

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

                pbar.update(1)

                if i % 50 == 49:
                    gc.collect()
                    if use_mps:
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()

        del model
        gc.collect()
        if use_mps:
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    [Emotion] Done — {total} segment(s), {errors} error(s), {total_inference_time:.1f}s inference")
        return transcript

    def _analyze_prosody(self, audio_path: Path, transcript: Transcript) -> Transcript:
        """Classify pitch trend, speech rate, and volume for each transcript entry.

        Pure librosa signal processing — no ML model, no GPU.
        Uses multiprocessing for parallel CPU utilization.
        """
        import multiprocessing as mp
        from tqdm import tqdm

        info = sf.info(str(audio_path))
        sr = info.samplerate
        max_samples = sr * 10

        total = len(transcript.entries)
        errors = 0

        # Prepare segment infos for parallel processing
        segments = []
        for i, entry in enumerate(transcript.entries):
            start_sample = int(entry.start * sr)
            end_sample = min(int(entry.end * sr), start_sample + max_samples)
            segments.append({
                "i": i,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration": entry.end - entry.start,
            })

        print(f"    [Prosody] Analyzing {total} segment(s) ...")
        
        # Use all CPU cores
        num_workers = mp.cpu_count()
        ctx = mp.get_context("spawn")
        
        pool = ctx.Pool(processes=num_workers)
        try:
            with tqdm(total=total, desc="    [Prosody]", unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for i, volume, speech_rate, pitch_trend, _, error in pool.imap(_prosody_segment_worker, [
                    (seg, str(audio_path), sr, max_samples) for seg in segments
                ]):
                    if error:
                        errors += 1
                        pbar.write(f"    [Prosody] Warning: entry {i} failed: {error}")
                    else:
                        if volume:
                            transcript.entries[i].volume = volume
                        if speech_rate:
                            transcript.entries[i].speech_rate = speech_rate
                        if pitch_trend:
                            transcript.entries[i].pitch_trend = pitch_trend
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n    [Prosody] Interrupted, terminating workers...")
            pool.terminate()
            pool.join()
            raise
        finally:
            pool.close()
            pool.join()

        print(f"    [Prosody] Done — {total} segment(s), {errors} error(s)")
        return transcript
