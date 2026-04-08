"""Speech-to-text runtime for `film.audio.stt`."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import soundfile as sf

from yorishiro.audio._speech_support import (
    TranscribeKwargs,
    assign_speaker,
    dummy_transcript_from_diarization,
    get_whisper_model,
    merge_speech_boundaries,
    split_text_heuristically,
    vad_chunk_boundaries,
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
    language: str | None = None


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
        out = output_dir / "transcript_raw.json"
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
        try:
            whisper_model = get_whisper_model(self.config)
        except ImportError:
            print("    [STT] faster-whisper not installed, using dummy transcript")
            return dummy_transcript_from_diarization(diarization, language)

        import librosa
        from tqdm import tqdm

        with sf.SoundFile(str(audio_path)) as audio_file:
            file_sample_rate = audio_file.samplerate
            total_duration = audio_file.frames / file_sample_rate

        boundaries = vad_chunk_boundaries(total_duration, cast(list[dict[str, float]], speech_segments))
        boundaries = merge_speech_boundaries(boundaries, cast(list[dict[str, float]], speech_segments))
        num_chunks = len(boundaries) - 1

        source_mtime = audio_path.stat().st_mtime
        checkpoint_dir = output_dir / ".stt_checkpoints" if output_dir else None
        if force and checkpoint_dir and checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)
            print("    [STT] Cleared checkpoints (force)")
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def transcribe_chunk(chunk_idx: int) -> tuple[int, list[TranscriptEntry], str | None]:
            chunk_start = boundaries[chunk_idx]
            chunk_end = boundaries[chunk_idx + 1]
            checkpoint_file = checkpoint_dir / f"chunk_{chunk_idx:04d}.json" if checkpoint_dir else None
            if checkpoint_file and checkpoint_file.exists():
                checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                if checkpoint.get("source_mtime") == source_mtime:
                    print(f"    [STT] chunk {chunk_idx + 1}/{num_chunks} — resuming from checkpoint")
                    return (
                        chunk_idx,
                        [TranscriptEntry(**entry) for entry in checkpoint["entries"]],
                        checkpoint.get("detected_language"),
                    )

            start_frame = int(chunk_start * file_sample_rate)
            end_frame = int(chunk_end * file_sample_rate)
            with sf.SoundFile(str(audio_path)) as handle:
                handle.seek(start_frame)
                chunk_audio = handle.read(end_frame - start_frame, dtype="float32")
            if getattr(chunk_audio, "ndim", 1) > 1:
                chunk_audio = chunk_audio.mean(axis=1)
            if file_sample_rate != 16000:
                chunk_audio = librosa.resample(chunk_audio, orig_sr=file_sample_rate, target_sr=16000)

            print(f"    [STT] chunk {chunk_idx + 1}/{num_chunks}  {chunk_start:.0f}s–{chunk_end:.0f}s ...", flush=True)
            transcribe_kwargs: TranscribeKwargs = {
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

            chunk_duration = chunk_end - chunk_start
            chunk_entries: list[TranscriptEntry] = []
            with tqdm(
                total=chunk_duration,
                desc=f"    [STT] chunk {chunk_idx + 1}/{num_chunks}",
                unit="s",
                bar_format="{l_bar}{bar}| {elapsed}<{remaining}",
            ) as progress:
                last_end = 0.0
                detected = getattr(info, "language", None)
                chunk_language = language or detected
                for segment in segments:
                    chunk_entries.extend(self._segment_to_entries(segment, chunk_start, diarization, chunk_language))
                    progress.update(min(segment.end, chunk_duration) - last_end)
                    last_end = min(segment.end, chunk_duration)

            if checkpoint_file:
                checkpoint_file.write_text(
                    json.dumps(
                        {
                            "chunk_idx": chunk_idx,
                            "source_mtime": source_mtime,
                            "entries": [entry.model_dump() for entry in chunk_entries],
                            "detected_language": detected,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return chunk_idx, chunk_entries, detected

        chunk_results: list[tuple[int, list[TranscriptEntry], str | None]] = []
        with ThreadPoolExecutor(max_workers=self.config.stt_num_workers) as pool:
            futures = [pool.submit(transcribe_chunk, idx) for idx in range(num_chunks)]
            for future in futures:
                chunk_results.append(future.result())

        chunk_results.sort(key=lambda chunk: chunk[0])
        entries: list[TranscriptEntry] = []
        detected_language: str | None = language
        for _, chunk_entries, chunk_language in chunk_results:
            entries.extend(chunk_entries)
            if detected_language is None and chunk_language:
                detected_language = chunk_language

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
        word_entries = self._split_segment_from_words(segment, chunk_start, diarization, confidence, language)
        if word_entries:
            return word_entries

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

        if len(timed_words) < 2:
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
