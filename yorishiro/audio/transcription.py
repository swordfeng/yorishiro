"""Speech-to-text runtime for `film.audio.stt`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import soundfile as sf
import torch

from yorishiro.audio import resample as audio_resample
from yorishiro.audio._speech_support import (
    TranscribeKwargs,
    get_transformers_pipeline,
    get_whisper_model,
    split_text_heuristically,
)
from yorishiro.models.film_models import STTEntry, STTTranscript

_STT_CHECKPOINT_FLUSH_GROUPS = 16


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
    stt_min_segment_seconds: float = 0.3
    stt_extra_args: dict[str, Any] = field(default_factory=dict)
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
    """Run STT using existing VAD output."""

    def __init__(self, config: TranscriberConfig | None = None) -> None:
        self.config = config or TranscriberConfig()

    def run(
        self,
        audio_path: Path,
        output_dir: Path,
        language: str | None = None,
        force: bool = False,
    ) -> STTTranscript:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "stt.json"
        if force and out.exists():
            out.unlink()
            print("    [STT] Cleared output (force)")
        vad_path = output_dir / "vad.json"
        detected_language = language or self.config.language
        input_signature = self._input_signature(
            audio_path,
            vad_path,
            detected_language,
        )
        checkpoint_dir = output_dir / ".stt_checkpoints"
        speech_segments = cast(
            list[dict[str, Any]],
            json.loads(vad_path.read_text(encoding="utf-8")),
        )
        print(f"  [STT] Transcribing {audio_path.name} ...")
        transcript = self._run_transcription(
            audio_path,
            speech_segments,
            detected_language,
            output_dir=output_dir,
            force=force,
            input_signature=input_signature,
        )
        try:
            self._assert_inputs_unchanged(
                audio_path,
                vad_path,
                detected_language,
                input_signature,
            )
        except RuntimeError:
            self._cleanup_checkpoint_dir(checkpoint_dir)
            raise
        self._atomic_write_text(out, transcript.model_dump_json(indent=2))
        self._cleanup_checkpoint_dir(checkpoint_dir)
        print(
            f"  [STT] Done — {len(transcript.entries)} segment(s), language: {transcript.language}"
        )
        return transcript

    def _run_transcription(
        self,
        audio_path: Path,
        speech_segments: list[dict[str, Any]],
        language: str | None,
        output_dir: Path | None = None,
        force: bool = False,
        input_signature: str | None = None,
    ) -> STTTranscript:
        from tqdm import tqdm

        with sf.SoundFile(str(audio_path)) as audio_file:
            file_sample_rate = audio_file.samplerate
            total_duration = audio_file.frames / file_sample_rate

        spans = self._speech_spans(cast(list[dict[str, float]], speech_segments), total_duration)
        groups = self._build_speech_groups(spans)
        if not groups:
            return STTTranscript(language=language or "unknown", entries=[])
        num_spans = len(spans)
        checkpoint_dir = (
            self._prepare_checkpoint_dir(output_dir, input_signature, force=force)
            if output_dir is not None and input_signature is not None
            else None
        )
        source_mtime = audio_path.stat().st_mtime
        completed = self._load_group_results(checkpoint_dir, source_mtime)
        pending_groups = [group for group in groups if group.group_id not in completed]

        progress_lock = threading.Lock()
        checkpoint_lock = threading.Lock()
        pending_checkpoint_results: list[GroupResult] = []
        completed_spans = sum(
            group.span_end_idx - group.span_start_idx + 1
            for group in groups
            if group.group_id in completed
        )

        if not pending_groups:
            print(f"    [STT] Resuming from checkpoints for {len(groups)} group(s)")
            return self._assemble_transcript(groups, completed, language)

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
                    pending_checkpoint_results.append(result)
                    if len(pending_checkpoint_results) >= _STT_CHECKPOINT_FLUSH_GROUPS:
                        self._flush_pending_checkpoint_results(
                            checkpoint_dir,
                            pending_checkpoint_results,
                        )

            try:
                new_results = self._transcribe_groups(
                    pending_groups,
                    audio_path=audio_path,
                    file_sample_rate=file_sample_rate,
                    language=language,
                    resample_module=audio_resample,
                    update_progress=update_progress,
                    save_result=save_result,
                )
            finally:
                if checkpoint_dir is not None:
                    with checkpoint_lock:
                        self._flush_pending_checkpoint_results(
                            checkpoint_dir,
                            pending_checkpoint_results,
                        )

        completed.update({result.group_id: result for result in new_results})
        return self._assemble_transcript(groups, completed, language)

    def _transcribe_groups(
        self,
        groups: list[SpeechGroup],
        *,
        audio_path: Path,
        file_sample_rate: int,
        language: str | None,
        resample_module: Any,
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
                language=language,
                resample_module=resample_module,
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
                    language=language,
                    resample_module=resample_module,
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
        language: str | None,
        resample_module: Any,
        update_progress: Any,
        save_result: Any,
    ) -> list[GroupResult]:
        if not groups:
            return []

        backend = worker_config.stt_backend
        if backend == "transformers-whisper":
            return self._run_worker_groups_transformers(
                groups,
                worker_idx=worker_idx,
                worker_config=worker_config,
                audio_path=audio_path,
                file_sample_rate=file_sample_rate,
                language=language,
                resample_module=resample_module,
                update_progress=update_progress,
                save_result=save_result,
            )

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
                chunk_audio = resample_module.resample(
                    chunk_audio, orig_sr=file_sample_rate, target_sr=16000
                )

            segments, info = whisper_model.transcribe(chunk_audio, **transcribe_kwargs)

            detected = getattr(info, "language", None)
            chunk_language = language or detected
            entries: list[dict[str, Any]] = []
            for segment in segments:
                for entry in self._segment_to_entries(
                    segment, group.start, group.end, chunk_language
                ):
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

    def _run_worker_groups_transformers(
        self,
        groups: list[SpeechGroup],
        *,
        worker_idx: int,
        worker_config: TranscriberConfig,
        audio_path: Path,
        file_sample_rate: int,
        language: str | None,
        resample_module: Any,
        update_progress: Any,
        save_result: Any,
    ) -> list[GroupResult]:
        pipe = get_transformers_pipeline(worker_config, instance_key=f"worker-{worker_idx}")
        generate_kwargs: dict[str, Any] = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        source_mtime = audio_path.stat().st_mtime
        results: list[GroupResult] = []
        for group in groups:
            start_frame = int(group.start * file_sample_rate)
            end_frame = int(group.end * file_sample_rate)
            chunk_audio, _ = sf.read(str(audio_path), start=start_frame, stop=end_frame, dtype="float32")
            if getattr(chunk_audio, "ndim", 1) > 1:
                chunk_audio = chunk_audio.mean(axis=1)
            if file_sample_rate != 16000:
                chunk_audio = resample_module.resample(
                    chunk_audio, orig_sr=file_sample_rate, target_sr=16000
                )

            group_confidence: float | None = None
            model = getattr(pipe, "model", None)
            if model is not None and hasattr(model, "generate"):
                captured_generate_outputs: list[Any] = []
                original_generate = model.generate

                def _capturing_generate(*args: Any, **kwargs: Any) -> Any:
                    original_kwargs = dict(kwargs)
                    kwargs["output_scores"] = True
                    kwargs["return_segments"] = True
                    kwargs["return_dict_in_generate"] = True
                    try:
                        generated = original_generate(*args, **kwargs)
                    except Exception:
                        generated = original_generate(*args, **original_kwargs)
                    captured_generate_outputs.append(generated)
                    sequences = self._extract_generate_sequences(generated)
                    return sequences if sequences is not None else generated

                try:
                    with patch.object(model, "generate", _capturing_generate):
                        output = pipe(
                            chunk_audio,
                            return_timestamps=True,
                            generate_kwargs=generate_kwargs if generate_kwargs else None,
                        )
                except Exception:
                    output = pipe(
                        chunk_audio,
                        return_timestamps=True,
                        generate_kwargs=generate_kwargs if generate_kwargs else None,
                    )
                group_confidence = self._transformers_group_confidence(
                    captured_generate_outputs
                )
            else:
                output = pipe(
                    chunk_audio,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs if generate_kwargs else None,
                )

            detected = output.get("language", language)
            chunk_language = language or detected
            entries: list[dict[str, Any]] = []
            for segment in self._transformers_chunks_to_segments(
                output.get("chunks", []),
                group.start,
                group.end,
                chunk_language,
                group_confidence=group_confidence,
            ):
                for entry in self._segment_to_entries(
                    segment, group.start, group.end, chunk_language
                ):
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

    @staticmethod
    def _transformers_chunks_to_segments(
        chunks: list[dict[str, Any]],
        chunk_start: float,
        chunk_end: float,
        language: str | None,
        *,
        group_confidence: float | None = None,
    ) -> list[Any]:
        """Convert transformers pipeline chunks into segment-like objects."""
        segments: list[Any] = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            timestamp = chunk.get("timestamp")
            if timestamp and isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                start = float(timestamp[0]) if timestamp[0] is not None else 0.0
                end = float(timestamp[1]) if timestamp[1] is not None else 0.0
            else:
                start = 0.0
                end = 0.0
            segments.append(
                SimpleNamespace(
                    start=start,
                    end=end,
                    text=text,
                    avg_logprob=group_confidence,
                    words=[],
                )
            )
        return segments

    @staticmethod
    def _extract_generate_segments(outputs: Any) -> list[Any]:
        segments = Transcriber._segment_value(outputs, "segments")
        if segments is None:
            return []
        if not isinstance(segments, list):
            return []
        if segments and isinstance(segments[0], list):
            first = segments[0]
            return first if isinstance(first, list) else []
        return segments

    @staticmethod
    def _extract_generate_sequences(outputs: Any) -> torch.Tensor | None:
        if isinstance(outputs, torch.Tensor):
            return outputs
        sequences = Transcriber._segment_value(outputs, "sequences")
        if isinstance(sequences, torch.Tensor):
            return sequences
        return None

    def _transformers_group_confidence(self, generate_outputs: list[Any]) -> float | None:
        scores: list[float] = []
        for generated in generate_outputs:
            segments = self._extract_generate_segments(generated)
            for segment in segments:
                segment_score = self._extract_segment_avg_logprob(segment)
                if segment_score is not None:
                    scores.append(segment_score)
            if segments:
                continue
            seq_scores = self._segment_value(generated, "sequences_scores")
            if isinstance(seq_scores, torch.Tensor):
                scores.extend(float(value.item()) for value in seq_scores.flatten())
            elif isinstance(seq_scores, (list, tuple)):
                scores.extend(
                    score for item in seq_scores if (score := self._as_float(item)) is not None
                )
            elif (score := self._as_float(seq_scores)) is not None:
                scores.append(score)
        if not scores:
            return None
        return sum(scores) / len(scores)

    @staticmethod
    def _segment_value(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        if hasattr(obj, key):
            return getattr(obj, key)
        try:
            return obj[key]
        except Exception:
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None

    def _extract_segment_avg_logprob(self, segment: Any) -> float | None:
        result = self._segment_value(segment, "result")
        if result is None:
            return None

        seq_scores = self._segment_value(result, "sequences_scores")
        if seq_scores is not None:
            if isinstance(seq_scores, torch.Tensor):
                if seq_scores.numel() == 0:
                    return None
                return float(seq_scores.flatten()[0].item())
            if isinstance(seq_scores, (list, tuple)) and seq_scores:
                return self._as_float(seq_scores[0])
            return self._as_float(seq_scores)

        scores = self._segment_value(result, "scores")
        sequences = self._segment_value(result, "sequences")
        idxs = self._segment_value(segment, "idxs")
        if not isinstance(scores, (list, tuple)) or sequences is None:
            return None
        if not isinstance(idxs, (list, tuple)) or len(idxs) < 2:
            return None

        if isinstance(sequences, torch.Tensor):
            if sequences.ndim == 0:
                return None
            sequence = sequences[0] if sequences.ndim > 1 else sequences
        else:
            return None

        gen_steps = len(scores)
        seq_len = int(sequence.shape[0]) if hasattr(sequence, "shape") else len(sequence)
        if gen_steps <= 0 or seq_len <= 0:
            return None

        score_offset = seq_len - gen_steps
        if score_offset < 0:
            return None

        start_idx = int(idxs[0])
        end_idx = int(idxs[1])
        if end_idx <= start_idx:
            return None

        local_start = max(0, start_idx - score_offset)
        local_end = min(gen_steps, end_idx - score_offset)
        if local_end <= local_start:
            return None

        token_positions = range(score_offset + local_start, score_offset + local_end)
        logprob_values: list[float] = []
        for score_idx, token_pos in enumerate(token_positions, start=local_start):
            logits = scores[score_idx]
            if not isinstance(logits, torch.Tensor):
                return None
            step_logits = logits[0] if logits.ndim > 1 else logits
            if not isinstance(step_logits, torch.Tensor):
                return None
            token_id = int(sequence[token_pos].item())
            token_logprob = torch.log_softmax(step_logits.float(), dim=-1)[token_id]
            logprob_values.append(float(token_logprob.item()))

        if not logprob_values:
            return None
        return sum(logprob_values) / len(logprob_values)

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
        min_dur = float(max(self.config.stt_min_segment_seconds, 0.0))
        for index, segment in enumerate(speech_segments):
            start = max(0.0, float(segment["start"]))
            raw_end = segment["end"]
            end = total_duration if raw_end == float("inf") else min(total_duration, float(raw_end))
            if end <= start:
                continue
            if (end - start) < min_dur:
                continue
            spans.append(SpeechSpan(index=len(spans), start=start, end=end))
        if spans:
            return spans
        # If VAD emitted nothing, fall back to whole file for compatibility.
        if not speech_segments:
            return [SpeechSpan(index=0, start=0.0, end=total_duration)]
        # VAD existed but all spans were filtered/invalid.
        return []

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

    def _prepare_checkpoint_dir(
        self,
        output_dir: Path,
        input_signature: str,
        *,
        force: bool = False,
    ) -> Path:
        checkpoint_dir = output_dir / ".stt_checkpoints"
        if force:
            had_checkpoints = checkpoint_dir.exists()
            self._cleanup_checkpoint_dir(checkpoint_dir)
            if had_checkpoints:
                print("    [STT] Cleared checkpoints (force)")
        elif checkpoint_dir.exists() and not self._checkpoint_matches(
            checkpoint_dir, input_signature
        ):
            self._cleanup_checkpoint_dir(checkpoint_dir)

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            self._write_checkpoint_metadata(checkpoint_dir, input_signature)
        return checkpoint_dir

    @staticmethod
    def _checkpoint_meta_path(checkpoint_dir: Path) -> Path:
        return checkpoint_dir / "meta.json"

    def _write_checkpoint_metadata(
        self,
        checkpoint_dir: Path,
        input_signature: str,
    ) -> None:
        payload = {"input_signature": input_signature}
        self._atomic_write_text(
            self._checkpoint_meta_path(checkpoint_dir),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _checkpoint_matches(self, checkpoint_dir: Path, input_signature: str) -> bool:
        meta_path = self._checkpoint_meta_path(checkpoint_dir)
        if not checkpoint_dir.exists() or not meta_path.exists():
            return False
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("input_signature") == input_signature

    @staticmethod
    def _cleanup_checkpoint_dir(checkpoint_dir: Path) -> None:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def _input_signature(
        self,
        audio_path: Path,
        vad_path: Path,
        language: str | None,
    ) -> str:
        audio_stat = audio_path.stat()
        vad_payload = vad_path.read_text(encoding="utf-8")
        payload = {
            "audio_path": str(audio_path.resolve()),
            "audio_mtime_ns": audio_stat.st_mtime_ns,
            "audio_size": audio_stat.st_size,
            "vad_sha256": hashlib.sha256(vad_payload.encode("utf-8")).hexdigest(),
            "language": language,
            "config": asdict(self.config),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _assert_inputs_unchanged(
        self,
        audio_path: Path,
        vad_path: Path,
        language: str | None,
        expected_signature: str,
    ) -> None:
        current_signature = self._input_signature(audio_path, vad_path, language)
        if current_signature != expected_signature:
            raise RuntimeError(
                "STT inputs changed during transcription; refusing to publish stale stt.json"
            )

    def _atomic_write_text(self, path: Path, content: str) -> None:
        fd, temp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

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
            "groups": [
                asdict(item)
                for item in sorted(
                    existing.values(), key=lambda item: item.span_start_idx
                )
            ]
        }
        self._atomic_write_text(
            shard_path,
            json.dumps(shard_payload, ensure_ascii=False, indent=2),
        )

    def _save_group_results_batch(
        self,
        checkpoint_dir: Path,
        results: list[GroupResult],
    ) -> None:
        if not results:
            return
        shard_updates: dict[int, dict[str, GroupResult]] = {}
        for result in results:
            shard_idx = result.span_start_idx // self.config.stt_checkpoint_shard_size
            updates = shard_updates.setdefault(shard_idx, {})
            updates[result.group_id] = result

        for shard_idx, updates in shard_updates.items():
            shard_path = checkpoint_dir / f"groups_{shard_idx:04d}.json"
            existing: dict[str, GroupResult] = {}
            if shard_path.exists():
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                for record in cast(list[dict[str, Any]], shard.get("groups", [])):
                    existing_result = GroupResult(**record)
                    existing[existing_result.group_id] = existing_result

            existing.update(updates)
            shard_payload = {
                "groups": [
                    asdict(item)
                    for item in sorted(
                        existing.values(), key=lambda item: item.span_start_idx
                    )
                ]
            }
            self._atomic_write_text(
                shard_path,
                json.dumps(shard_payload, ensure_ascii=False, indent=2),
            )

    def _flush_pending_checkpoint_results(
        self,
        checkpoint_dir: Path,
        pending_results: list[GroupResult],
    ) -> None:
        if not pending_results:
            return
        self._save_group_results_batch(checkpoint_dir, pending_results)
        pending_results.clear()

    def _assemble_transcript(
        self,
        groups: list[SpeechGroup],
        group_results: dict[str, GroupResult],
        language: str | None,
    ) -> STTTranscript:
        entries: list[STTEntry] = []
        detected_language = language
        for group in groups:
            result = group_results.get(group.group_id)
            if result is None:
                continue
            entries.extend(STTEntry(**entry) for entry in result.entries)
            if detected_language is None and result.detected_language:
                detected_language = result.detected_language
        entries.sort(key=lambda entry: (entry.start, entry.end))
        for idx, entry in enumerate(entries):
            entry.entry_id = f"utt_{idx:06d}"
        return STTTranscript(language=detected_language or "unknown", entries=entries)

    def _segment_to_entries(
        self,
        segment: Any,
        chunk_start: float,
        chunk_end: float,
        language: str | None,
    ) -> list[STTEntry]:
        text = segment.text.strip()
        if not text:
            return []

        raw_confidence = getattr(segment, "avg_logprob", None)
        raw_confidence = getattr(segment, "avg_logprob", None)
        confidence = raw_confidence if raw_confidence is not None else 0.0
        abs_start = chunk_start + float(segment.start)
        abs_end = chunk_start + float(segment.end)
        clamped_start = max(abs_start, chunk_start)
        clamped_end = min(abs_end, chunk_end)
        duration = max(clamped_end - clamped_start, 0.0)
        chars_per_second = (len(text) / duration) if duration > 0 else float("inf")
        if raw_confidence is not None and confidence < self.config.stt_min_confidence:
            return []
        if duration < float(max(self.config.stt_min_segment_seconds, 0.0)):
            return []
        if chars_per_second > self.config.stt_max_chars_per_second:
            return []
        return [
            STTEntry(
                entry_id="",
                start=clamped_start,
                end=clamped_end,
                text=text,
                confidence=confidence,
            )
        ]

    def _split_segment_from_words(
        self,
        segment: Any,
        chunk_start: float,
        confidence: float,
        language: str | None,
    ) -> list[STTEntry]:
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

        entries: list[STTEntry] = []
        for group in groups:
            group_text = "".join(token for _, _, token in group).strip()
            if not group_text:
                continue
            abs_start = chunk_start + group[0][0]
            abs_end = chunk_start + group[-1][1]
            entries.extend(self._split_entry_text(abs_start, abs_end, group_text, confidence, language))
        return entries

    def _split_entry_text(
        self,
        start: float,
        end: float,
        text: str,
        confidence: float,
        language: str | None,
    ) -> list[STTEntry]:
        pieces = split_text_heuristically(text, language)
        if len(pieces) <= 1:
            return [
                STTEntry(
                    entry_id="",
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                )
            ]

        total_chars = sum(len(piece) for piece in pieces)
        duration = max(end - start, 0.0)
        cursor = start
        entries: list[STTEntry] = []
        for idx, piece in enumerate(pieces):
            piece_duration = duration * (len(piece) / total_chars) if total_chars else 0.0
            piece_end = end if idx == len(pieces) - 1 else min(end, cursor + piece_duration)
            entries.append(
                STTEntry(
                    entry_id="",
                    start=cursor,
                    end=piece_end,
                    text=piece,
                    confidence=confidence,
                )
            )
            cursor = piece_end
        return entries
