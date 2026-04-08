"""Speech-to-text runtime for `film.audio.stt`."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import soundfile as sf
import torch

from yorishiro.audio._speech_support import (
    TranscribeKwargs,
    assign_speaker,
    get_whisper_model,
    split_text_heuristically,
)
from yorishiro.models.film_models import Transcript, TranscriptEntry


@dataclass(frozen=True)
class TranscriberConfig:
    stt_backend: str = "faster-whisper"
    stt_model: str = "large-v3"
    stt_cpu_threads: int = 0
    stt_num_workers: int = 1
    stt_word_timestamps: bool = False
    stt_vad_filter: bool = False
    stt_vad_min_silence_duration_ms: int = 500
    stt_checkpoint_shard_size: int = 500
    stt_group_max_duration_seconds: float = 30.0
    stt_group_max_gap_seconds: float = 0.6
    stt_min_confidence: float = -0.5
    stt_max_chars_per_second: float = 28.0
    language: str | None = None


@dataclass(frozen=True)
class SpeechSpan:
    index: int
    start: float
    end: float


@dataclass(frozen=True)
class SpeechGroup:
    group_id: str
    span_start_idx: int
    span_end_idx: int
    start: float
    end: float


@dataclass(frozen=True)
class GroupResult:
    group_id: str
    span_start_idx: int
    span_end_idx: int
    start: float
    end: float
    entries: list[dict[str, Any]]
    detected_language: str | None
    source_mtime: float


class Transcriber:
    """Run STT using existing VAD and diarization outputs."""

    def __init__(self, config: TranscriberConfig | None = None) -> None:
        self.config = config or TranscriberConfig()

    def run(
        self,
        audio_path: Path,
        output_dir: Path,
        language: str | None = None,
        force: bool = False,
    ) -> Transcript:
        output_dir.mkdir(parents=True, exist_ok=True)
        vad_path = output_dir / "vad.json"
        diar_path = output_dir / "diarization.json"
        speech_segments = cast(list[dict[str, Any]], json.loads(vad_path.read_text(encoding="utf-8")))
        diarization = cast(list[dict[str, Any]], json.loads(diar_path.read_text(encoding="utf-8")))
        detected_language = language or self.config.language
        print(f"  [STT] Transcribing {audio_path.name} ...")
        transcript = self._run_transcription(
            audio_path,
            diarization,
            speech_segments,
            detected_language,
            output_dir=output_dir,
            force=force,
        )
        out = output_dir / "transcription.json"
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [STT] Done — {len(transcript.entries)} segment(s), language: {transcript.language}")
        return transcript

    def _run_transcription(
        self,
        audio_path: Path,
        diarization: list[dict[str, Any]],
        speech_segments: list[dict[str, Any]],
        language: str | None,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> Transcript:
        import librosa
        from tqdm import tqdm

        with sf.SoundFile(str(audio_path)) as audio_file:
            file_sample_rate = audio_file.samplerate
            total_duration = audio_file.frames / file_sample_rate

        spans = self._speech_spans(cast(list[dict[str, float]], speech_segments), total_duration)
        groups = self._build_speech_groups(spans)
        num_spans = len(spans)
        source_mtime = audio_path.stat().st_mtime
        checkpoint_dir = output_dir / ".stt_checkpoints" if output_dir else None

        if force and checkpoint_dir and checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)
            print("    [STT] Cleared checkpoints (force)")
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        completed = self._load_group_results(checkpoint_dir, source_mtime)
        pending_groups = [group for group in groups if group.group_id not in completed]

        if not pending_groups:
            print(f"    [STT] Resuming from checkpoints for {len(groups)} group(s)")
            return self._assemble_transcript(groups, completed, language)

        progress_lock = threading.Lock()
        checkpoint_lock = threading.Lock()
        completed_spans = sum(group.span_end_idx - group.span_start_idx + 1 for group in groups if group.group_id in completed)

        with tqdm(
            total=num_spans,
            desc="    [STT] VAD spans",
            unit="span",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            initial=completed_spans,
        ) as progress:
            def update_progress(delta: int) -> None:
                with progress_lock:
                    progress.update(delta)

            def save_result(result: GroupResult) -> None:
                if checkpoint_dir is None:
                    return
                with checkpoint_lock:
                    self._save_group_result(checkpoint_dir, result)

            new_results = self._transcribe_groups(
                pending_groups,
                audio_path=audio_path,
                file_sample_rate=file_sample_rate,
                diarization=diarization,
                language=language,
                librosa_module=librosa,
                update_progress=update_progress,
                save_result=save_result,
            )

        completed.update({result.group_id: result for result in new_results})
        return self._assemble_transcript(groups, completed, language)

    def _transcribe_groups(
        self,
        groups: list[SpeechGroup],
        *,
        audio_path: Path,
        file_sample_rate: int,
        diarization: list[dict[str, Any]],
        language: str | None,
        librosa_module: Any,
        update_progress: Any,
        save_result: Any,
    ) -> list[GroupResult]:
        if not groups:
            return []

        if torch.cuda.is_available():
            worker_groups = [groups]
            worker_total = 1
        else:
            worker_total = max(1, min(self.config.stt_num_workers, len(groups)))
            worker_groups = [groups[idx::worker_total] for idx in range(worker_total)]

        worker_configs = self._worker_configs(worker_total)

        if worker_total == 1:
            return self._run_worker_groups(
                worker_groups[0],
                worker_idx=0,
                worker_config=worker_configs[0],
                audio_path=audio_path,
                file_sample_rate=file_sample_rate,
                diarization=diarization,
                language=language,
                librosa_module=librosa_module,
                update_progress=update_progress,
                save_result=save_result,
            )

        results: list[GroupResult] = []
        with ThreadPoolExecutor(max_workers=worker_total) as pool:
            futures = [
                pool.submit(
                    self._run_worker_groups,
                    worker_group,
                    worker_idx=worker_idx,
                    worker_config=worker_configs[worker_idx],
                    audio_path=audio_path,
                    file_sample_rate=file_sample_rate,
                    diarization=diarization,
                    language=language,
                    librosa_module=librosa_module,
                    update_progress=update_progress,
                    save_result=save_result,
                )
                for worker_idx, worker_group in enumerate(worker_groups)
                if worker_group
            ]
            for future in futures:
                results.extend(future.result())
        return results

    def _run_worker_groups(
        self,
        groups: list[SpeechGroup],
        *,
        worker_idx: int,
        worker_config: TranscriberConfig,
        audio_path: Path,
        file_sample_rate: int,
        diarization: list[dict[str, Any]],
        language: str | None,
        librosa_module: Any,
        update_progress: Any,
        save_result: Any,
    ) -> list[GroupResult]:
        if not groups:
            return []

        whisper_model = get_whisper_model(worker_config, instance_key=f"worker-{worker_idx}")
        transcribe_kwargs: TranscribeKwargs = {
            "language": language or None,
            "task": "transcribe",
            "vad_filter": worker_config.stt_vad_filter,
            "word_timestamps": worker_config.stt_word_timestamps,
            "condition_on_previous_text": False,
        }
        if worker_config.stt_vad_filter:
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": worker_config.stt_vad_min_silence_duration_ms,
            }

        source_mtime = audio_path.stat().st_mtime
        results: list[GroupResult] = []
        for group in groups:
            start_frame = int(group.start * file_sample_rate)
            end_frame = int(group.end * file_sample_rate)
            chunk_audio, _ = sf.read(str(audio_path), start=start_frame, stop=end_frame, dtype="float32")
            if getattr(chunk_audio, "ndim", 1) > 1:
                chunk_audio = chunk_audio.mean(axis=1)
            if file_sample_rate != 16000:
                chunk_audio = librosa_module.resample(chunk_audio, orig_sr=file_sample_rate, target_sr=16000)

            segments, info = whisper_model.transcribe(chunk_audio, **transcribe_kwargs)

            detected = getattr(info, "language", None)
            chunk_language = language or detected
            entries: list[dict[str, Any]] = []
            for segment in segments:
                for entry in self._segment_to_entries(segment, group.start, diarization, chunk_language):
                    entries.append(entry.model_dump())
            results.append(
                GroupResult(
                    group_id=group.group_id,
                    span_start_idx=group.span_start_idx,
                    span_end_idx=group.span_end_idx,
                    start=group.start,
                    end=group.end,
                    entries=entries,
                    detected_language=detected,
                    source_mtime=source_mtime,
                )
            )
            save_result(results[-1])
            update_progress(group.span_end_idx - group.span_start_idx + 1)
        return results

    def _worker_configs(self, worker_total: int) -> list[TranscriberConfig]:
        if torch.cuda.is_available():
            return [replace(self.config, stt_num_workers=1)]

        total_threads = self.config.stt_cpu_threads or os.cpu_count() or 4
        threads_per_worker = max(1, total_threads // worker_total)
        return [
            replace(self.config, stt_cpu_threads=threads_per_worker, stt_num_workers=1)
            for _ in range(worker_total)
        ]

    def _speech_spans(
        self,
        speech_segments: list[dict[str, float]],
        total_duration: float,
    ) -> list[SpeechSpan]:
        spans: list[SpeechSpan] = []
        for index, segment in enumerate(speech_segments):
            start = max(0.0, float(segment["start"]))
            raw_end = segment["end"]
            end = total_duration if raw_end == float("inf") else min(total_duration, float(raw_end))
            if end <= start:
                continue
            spans.append(SpeechSpan(index=len(spans), start=start, end=end))
        return spans or [SpeechSpan(index=0, start=0.0, end=total_duration)]

    def _build_speech_groups(self, spans: list[SpeechSpan]) -> list[SpeechGroup]:
        if not spans:
            return []

        groups: list[SpeechGroup] = []
        current = [spans[0]]
        for span in spans[1:]:
            prev = current[-1]
            candidate_duration = span.end - current[0].start
            gap = span.start - prev.end
            if (
                gap <= self.config.stt_group_max_gap_seconds
                and candidate_duration <= self.config.stt_group_max_duration_seconds
            ):
                current.append(span)
                continue
            groups.append(self._speech_group_from_spans(current))
            current = [span]
        groups.append(self._speech_group_from_spans(current))
        return groups

    def _speech_group_from_spans(self, spans: list[SpeechSpan]) -> SpeechGroup:
        first = spans[0]
        last = spans[-1]
        return SpeechGroup(
            group_id=f"g_{first.index:06d}_{last.index:06d}",
            span_start_idx=first.index,
            span_end_idx=last.index,
            start=first.start,
            end=last.end,
        )

    def _load_group_results(
        self,
        checkpoint_dir: Path | None,
        source_mtime: float,
    ) -> dict[str, GroupResult]:
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return {}

        results: dict[str, GroupResult] = {}
        for shard_path in sorted(checkpoint_dir.glob("groups_*.json")):
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            for record in cast(list[dict[str, Any]], shard.get("groups", [])):
                if record.get("source_mtime") != source_mtime:
                    continue
                result = GroupResult(**record)
                results[result.group_id] = result
        return results

    def _save_group_result(self, checkpoint_dir: Path, result: GroupResult) -> None:
        shard_idx = result.span_start_idx // self.config.stt_checkpoint_shard_size
        shard_path = checkpoint_dir / f"groups_{shard_idx:04d}.json"
        existing: dict[str, GroupResult] = {}
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            for record in cast(list[dict[str, Any]], shard.get("groups", [])):
                existing_result = GroupResult(**record)
                existing[existing_result.group_id] = existing_result

        existing[result.group_id] = result
        shard_payload = {
            "groups": [asdict(item) for item in sorted(existing.values(), key=lambda item: item.span_start_idx)]
        }
        shard_path.write_text(json.dumps(shard_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _assemble_transcript(
        self,
        groups: list[SpeechGroup],
        group_results: dict[str, GroupResult],
        language: str | None,
    ) -> Transcript:
        entries: list[TranscriptEntry] = []
        detected_language = language
        for group in groups:
            result = group_results.get(group.group_id)
            if result is None:
                continue
            entries.extend(TranscriptEntry(**entry) for entry in result.entries)
            if detected_language is None and result.detected_language:
                detected_language = result.detected_language
        entries.sort(key=lambda entry: (entry.start, entry.end))
        return Transcript(language=detected_language or "unknown", entries=entries)

    def _segment_to_entries(
        self,
        segment: Any,
        chunk_start: float,
        diarization: list[dict[str, Any]],
        language: str | None,
    ) -> list[TranscriptEntry]:
        text = segment.text.strip()
        if not text:
            return []

        confidence = segment.avg_logprob if hasattr(segment, "avg_logprob") else 0.9
        duration = max(float(segment.end) - float(segment.start), 0.0)
        chars_per_second = (len(text) / duration) if duration > 0 else float("inf")
        if confidence < self.config.stt_min_confidence:
            return []
        if chars_per_second > self.config.stt_max_chars_per_second:
            return []
        abs_start = chunk_start + segment.start
        abs_end = chunk_start + segment.end
        return [
            TranscriptEntry(
                speaker_global=assign_speaker(abs_start, abs_end, diarization),
                start=abs_start,
                end=abs_end,
                text=text,
                confidence=confidence,
            )
        ]

    def _split_segment_from_words(
        self,
        segment: Any,
        chunk_start: float,
        diarization: list[dict[str, Any]],
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

        if len(timed_words) <= 1:
            return []

        groups: list[list[tuple[float, float, str]]] = []
        current = [timed_words[0]]
        for prev, cur in zip(timed_words, timed_words[1:]):
            prev_end = prev[1]
            cur_start, _, _ = cur
            pause = cur_start - prev_end
            should_split = pause >= 0.35
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
        diarization: list[dict[str, Any]],
        language: str | None,
    ) -> list[TranscriptEntry]:
        pieces = split_text_heuristically(text, language)
        if len(pieces) <= 1:
            return [
                TranscriptEntry(
                    speaker_global=assign_speaker(start, end, diarization),
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
            entries.append(
                TranscriptEntry(
                    speaker_global=assign_speaker(cursor, piece_end, diarization),
                    start=cursor,
                    end=piece_end,
                    text=piece,
                    confidence=confidence,
                )
            )
            cursor = piece_end
        return entries
