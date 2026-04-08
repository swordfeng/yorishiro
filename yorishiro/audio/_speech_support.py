"""Shared helpers for stage-native speech processing runtimes."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from yorishiro.models.film_models import Transcript, TranscriptEntry
from yorishiro.utils import get_device

_DIARIZATION_CHUNK_SECONDS = 1200.0
_SPEAKER_SIM_THRESHOLD = 0.75
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


class TranscribeKwargs(TypedDict):
    language: str | None
    task: Literal["transcribe"]
    vad_filter: bool
    word_timestamps: bool
    condition_on_previous_text: bool
    vad_parameters: NotRequired[dict[str, int]]


class DiarizerConfigLike(Protocol):
    diarization_model: str
    diarization_batch_size: int
    hf_token_env: str


class TranscriberConfigLike(Protocol):
    stt_model: str
    stt_cpu_threads: int
    stt_num_workers: int
    stt_word_timestamps: bool
    stt_vad_filter: bool
    stt_vad_min_silence_duration_ms: int


class EmotionAnalyzerConfigLike(Protocol):
    emotion_model: str


class DiarizationPipelineLike(Protocol):
    def __call__(self, path: str, hook: Any) -> Any: ...


class WhisperModelLike(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class EmotionModelLike(Protocol):
    def generate(self, **kwargs: Any) -> Any: ...


_DIARIZATION_PIPELINES: dict[tuple[str, int, str], DiarizationPipelineLike] = {}
_WHISPER_MODELS: dict[tuple[str, int, int], WhisperModelLike] = {}
_EMOTION_MODELS: dict[tuple[str, str], EmotionModelLike] = {}


def get_diarization_pipeline(config: DiarizerConfigLike) -> DiarizationPipelineLike:
    from pyannote.audio import Pipeline

    hf_token = os.environ.get(config.hf_token_env)
    if not hf_token:
        raise RuntimeError(f"{config.hf_token_env} not set")

    key = (config.diarization_model, config.diarization_batch_size, get_device())
    cached = _DIARIZATION_PIPELINES.get(key)
    if cached is not None:
        return cached

    pipeline = Pipeline.from_pretrained(config.diarization_model, token=hf_token)
    device = torch.device(get_device())
    pipeline = pipeline.to(device)  # type: ignore[union-attr]  # ty:ignore[unresolved-attribute]
    if hasattr(pipeline, "_segmentation"):
        pipeline._segmentation.batch_size = config.diarization_batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = config.diarization_batch_size
    _DIARIZATION_PIPELINES[key] = cast(DiarizationPipelineLike, pipeline)
    return cast(DiarizationPipelineLike, pipeline)


def get_whisper_model(config: TranscriberConfigLike) -> WhisperModelLike:
    from faster_whisper import WhisperModel

    cpu_threads = config.stt_cpu_threads or os.cpu_count() or 4
    num_workers = config.stt_num_workers
    key = (config.stt_model, cpu_threads, num_workers)
    cached = _WHISPER_MODELS.get(key)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        config.stt_model,
        device=device,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )
    print(f"    [STT] Loaded {config.stt_model} on {device} ({cpu_threads} threads x {num_workers} workers)")
    _WHISPER_MODELS[key] = cast(WhisperModelLike, model)
    return cast(WhisperModelLike, model)


def get_emotion_model(config: EmotionAnalyzerConfigLike) -> EmotionModelLike:
    from funasr import AutoModel

    device = get_device()
    key = (config.emotion_model, device)
    cached = _EMOTION_MODELS.get(key)
    if cached is not None:
        return cached

    model = AutoModel(
        model=config.emotion_model,
        hub="hf",
        disable_update=True,
        device=device,
    )
    _EMOTION_MODELS[key] = cast(EmotionModelLike, model)
    return cast(EmotionModelLike, model)


def vad_chunk_boundaries(
    total_duration: float,
    vad_segments: list[dict[str, float]],
    chunk_seconds: float = _DIARIZATION_CHUNK_SECONDS,
) -> list[float]:
    """Chunk boundaries snapped to silence gaps."""
    num_chunks = max(1, math.ceil(total_duration / chunk_seconds))
    if num_chunks == 1:
        return [0.0, total_duration]

    gap_mids: list[float] = []
    if vad_segments:
        if vad_segments[0]["start"] > 0:
            gap_mids.append(vad_segments[0]["start"] / 2)
        for idx in range(len(vad_segments) - 1):
            gap_mids.append((vad_segments[idx]["end"] + vad_segments[idx + 1]["start"]) / 2)
        if vad_segments[-1]["end"] < total_duration:
            gap_mids.append((vad_segments[-1]["end"] + total_duration) / 2)

    boundaries: list[float] = [0.0]
    for idx in range(1, num_chunks):
        target = idx * (total_duration / num_chunks)
        boundaries.append(min(gap_mids, key=lambda mid: abs(mid - target)) if gap_mids else target)
    boundaries.append(total_duration)

    seen: set[float] = set()
    deduped: list[float] = []
    for boundary in boundaries:
        if boundary not in seen:
            deduped.append(boundary)
            seen.add(boundary)
    return deduped


def merge_speech_boundaries(
    boundaries: list[float],
    speech_segments: list[dict[str, float]],
) -> list[float]:
    """Merge pure-silence chunks into a neighboring chunk."""

    def has_speech(start: float, end: float) -> bool:
        return any(seg["start"] < end and seg["end"] > start for seg in speech_segments)

    merged = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        if has_speech(merged[-1], boundary):
            merged.append(boundary)
    merged.append(boundaries[-1])
    while len(merged) > 2 and not has_speech(merged[-2], merged[-1]):
        merged.pop(-2)
    return merged


def assign_speaker(start: float, end: float, diarization: list[dict[str, Any]]) -> str:
    """Find the diarization speaker with the greatest overlap with [start, end]."""
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0
    for seg in diarization:
        seg_end = seg["end"] if seg["end"] != float("inf") else end + 1.0
        overlap = min(end, seg_end) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = cast(str, seg["speaker"])
    return best_speaker


def normalize_language(language: str | None) -> str | None:
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


def split_japanese_clause_text(text: str) -> list[str]:
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


def split_text_heuristically(text: str, language: str | None) -> list[str]:
    pieces = [part.strip() for part in _STT_PUNCT_SPLIT_RE.split(text) if part.strip()]
    if len(pieces) > 1:
        return pieces
    if len(text) < _STT_TEXT_SPLIT_MIN_CHARS:
        return [text]
    normalized_language = normalize_language(language)
    if normalized_language == "ja":
        return split_japanese_clause_text(text)
    return [text]


def merge_chunk_speakers(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Merge per-chunk speaker labels into global IDs via clustering."""
    all_embeddings: list[np.ndarray] = []
    metadata: list[tuple[int, str]] = []

    for chunk_idx, chunk in enumerate(chunks):
        embeddings = cast(np.ndarray | None, chunk["embeddings"])
        speakers_local = cast(list[str], chunk["speakers_local"])
        if embeddings is not None:
            for index, local_id in enumerate(speakers_local):
                all_embeddings.append(embeddings[index])
                metadata.append((chunk_idx, local_id))

    if len(all_embeddings) == 0:
        all_turns: list[dict[str, Any]] = []
        for chunk_idx, chunk in enumerate(chunks):
            for turn in cast(list[dict[str, Any]], chunk["turns"]):
                all_turns.append({**turn, "speaker": f"SPEAKER_{chunk_idx:02d}_{turn['speaker']}"})
        all_turns.sort(key=lambda turn: cast(float, turn["start"]))
        return all_turns, {}

    X = np.stack(all_embeddings)
    distances = pdist(X, metric="cosine")
    Z = linkage(distances, method="average")
    threshold = 1.0 - _SPEAKER_SIM_THRESHOLD
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")
    unique_clusters = np.unique(cluster_labels)
    cluster_to_speaker = {cluster: f"SPEAKER_{idx:02d}" for idx, cluster in enumerate(unique_clusters)}

    local_to_global: dict[tuple[int, str], str] = {}
    for (chunk_idx, local_id), cluster_label in zip(metadata, cluster_labels):
        local_to_global[(chunk_idx, local_id)] = cluster_to_speaker[int(cluster_label)]

    speaker_emb_accum: dict[str, list[np.ndarray]] = {}
    for emb, cluster_label in zip(all_embeddings, cluster_labels):
        global_id = cluster_to_speaker[int(cluster_label)]
        speaker_emb_accum.setdefault(global_id, []).append(emb)
    speaker_embeddings = {
        speaker_id: np.mean(np.stack(embeddings), axis=0)
        for speaker_id, embeddings in speaker_emb_accum.items()
    }

    print(
        f"    [Diarization] Clustered {len(all_embeddings)} local speakers into {len(unique_clusters)} global speakers"
    )

    all_turns: list[dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunks):
        for turn in cast(list[dict[str, Any]], chunk["turns"]):
            global_speaker = local_to_global.get((chunk_idx, cast(str, turn["speaker"])), cast(str, turn["speaker"]))
            all_turns.append({**turn, "speaker": global_speaker})

    all_turns.sort(key=lambda turn: cast(float, turn["start"]))
    return all_turns, speaker_embeddings


def save_speaker_bank(
    output_dir: Path,
    segments: list[dict[str, Any]],
    speaker_embeddings: dict[str, np.ndarray],
) -> None:
    from yorishiro.audio.speaker_bank import SpeakerBankManager

    bank = SpeakerBankManager()
    speakers = sorted({cast(str, turn["speaker"]) for turn in segments})
    first_appearance: dict[str, float] = {}
    for turn in segments:
        speaker = cast(str, turn["speaker"])
        if speaker not in first_appearance:
            first_appearance[speaker] = cast(float, turn["start"])

    for speaker_id in speakers:
        bank.speaker_bank.add_speaker(speaker_id, first_appearance.get(speaker_id, 0.0))
        if speaker_id in speaker_embeddings:
            bank._embeddings[speaker_id] = speaker_embeddings[speaker_id]

    bank.save(output_dir)
    print(f"    [Diarization] Speaker bank saved with {len(speakers)} speaker(s)")


def prosody_segment_worker(args: tuple[dict[str, int | float], str, int, int]) -> tuple[int, str | None, str | None, str | None, None, str | None]:
    """Worker for multiprocessing prosody analysis."""
    idx_info, audio_path_str, sr, _max_samples = args

    index = int(idx_info["i"])
    start_sample = int(idx_info["start_sample"])
    end_sample = int(idx_info["end_sample"])

    if end_sample - start_sample < int(sr * 0.1):
        return (index, None, None, None, None, None)

    import librosa

    try:
        chunk, _ = sf.read(audio_path_str, start=start_sample, stop=end_sample, dtype="float32")
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        volume: str | None = None
        speech_rate: str | None = None
        pitch_trend: str | None = None

        rms_frames = librosa.feature.rms(y=chunk)[0]
        db = 20.0 * np.log10(float(np.mean(rms_frames)) + 1e-9)
        volume = "quiet" if db < -38.0 else ("loud" if db > -20.0 else "normal")

        duration = float(idx_info["duration"])
        if duration >= 0.3:
            onsets = librosa.onset.onset_detect(y=chunk, sr=sr, units="time", normalize=True)
            rate = len(onsets) / duration
            speech_rate = "slow" if rate < 2.0 else ("fast" if rate > 4.0 else "normal")

        f0, voiced_flag, _ = librosa.pyin(
            chunk,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
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

        return (index, volume, speech_rate, pitch_trend, None, None)
    except Exception as exc:
        return (index, None, None, None, None, str(exc))


def dummy_transcript_from_diarization(
    diarization: list[dict[str, Any]],
    language: str | None,
) -> Transcript:
    entries = [
        TranscriptEntry(
            speaker_global=cast(str, seg["speaker"]),
            start=0.0 if cast(float, seg["start"]) == float("inf") else cast(float, seg["start"]),
            end=0.0 if cast(float, seg["end"]) == float("inf") else cast(float, seg["end"]),
            text="[Transcription unavailable]",
            confidence=0.0,
        )
        for seg in diarization
        if cast(float, seg["start"]) != float("inf")
    ]
    return Transcript(language=language or "unknown", entries=entries)
