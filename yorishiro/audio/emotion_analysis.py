"""Emotion and prosody runtime for `film.audio.emotion`."""

from __future__ import annotations

import contextlib
import gc
import io
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from yorishiro.audio._speech_support import get_emotion_model, prosody_segment_worker
from yorishiro.models.film_models import Transcript


@dataclass(frozen=True)
class EmotionAnalyzerConfig:
    emotion_backend: str = "emotion2vec"
    emotion_model: str = "emotion2vec/emotion2vec_plus_base"


class EmotionAnalyzer:
    """Run emotion and prosody enrichment on `transcript_raw.json`."""

    def __init__(self, config: EmotionAnalyzerConfig | None = None) -> None:
        self.config = config or EmotionAnalyzerConfig()

    def run(self, audio_path: Path, output_dir: Path) -> Transcript:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "transcript_raw.json"
        transcript = Transcript(**json.loads(raw_path.read_text(encoding="utf-8")))
        print(f"  [Emotion] Analyzing {len(transcript.entries)} segment(s) ...")
        transcript = self._analyze_emotions(audio_path, transcript)
        gc.collect()
        print(f"  [Prosody] Analyzing {len(transcript.entries)} segment(s) ...")
        transcript = self._analyze_prosody(audio_path, transcript)
        emotions = {entry.emotion for entry in transcript.entries if entry.emotion}
        out = output_dir / "transcript.json"
        out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        print(f"  [Emotion] Done — {', '.join(sorted(emotions)) if emotions else 'none'}")
        return transcript

    def _analyze_emotions(self, audio_path: Path, transcript: Transcript) -> Transcript:
        info = sf.info(str(audio_path))
        file_sr = info.samplerate
        target_sr = 16000
        max_samples = int(file_sr * 10)

        total = len(transcript.entries)
        errors = 0
        total_inference_time = 0.0

        print(f"    [Emotion] Loading model on {self._device_string()} ...")
        model = get_emotion_model(self.config)

        need_resample = file_sr != target_sr
        if need_resample:
            import librosa

        print(f"    [Emotion] Analyzing {total} segment(s) (file_sr={file_sr}) ...")
        with tqdm(
            total=total,
            desc="    [Emotion]",
            unit="seg",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as progress:
            for idx, entry in enumerate(transcript.entries):
                seg_duration = entry.end - entry.start
                progress.set_postfix_str(f"seg={seg_duration:.1f}s")

                start_sample = int(entry.start * file_sr)
                end_sample = min(int(entry.end * file_sr), start_sample + max_samples)
                if end_sample - start_sample < int(file_sr * 0.1):
                    progress.update(1)
                    continue

                try:
                    chunk, _ = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")
                except Exception as exc:
                    errors += 1
                    progress.write(f"    [Emotion] Warning: entry {idx} read error: {exc}")
                    progress.update(1)
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
                        best_idx = int(max(range(len(scores)), key=lambda score_idx: scores[score_idx]))
                        raw_label = labels[best_idx]
                        entry.emotion = raw_label.split("/")[-1] if "/" in raw_label else raw_label
                        entry.confidence = max(entry.confidence, scores[best_idx])
                except Exception as exc:
                    errors += 1
                    progress.write(f"    [Emotion] Warning: entry {idx} failed: {exc}")

                progress.update(1)

                if idx % 50 == 49:
                    gc.collect()
                    self._clear_torch_cache()

        print(f"    [Emotion] Done — {total} segment(s), {errors} error(s), {total_inference_time:.1f}s inference")
        return transcript

    def _analyze_prosody(self, audio_path: Path, transcript: Transcript) -> Transcript:
        info = sf.info(str(audio_path))
        sample_rate = info.samplerate
        max_samples = sample_rate * 10

        segments: list[dict[str, int | float]] = []
        for idx, entry in enumerate(transcript.entries):
            start_sample = int(entry.start * sample_rate)
            end_sample = min(int(entry.end * sample_rate), start_sample + max_samples)
            segments.append(
                {
                    "i": idx,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "duration": entry.end - entry.start,
                }
            )

        errors = 0
        total = len(transcript.entries)
        print(f"    [Prosody] Analyzing {total} segment(s) ...")

        num_workers = mp.cpu_count()
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=num_workers)
        try:
            with tqdm(
                total=total,
                desc="    [Prosody]",
                unit="seg",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as progress:
                for idx, volume, speech_rate, pitch_trend, _, error in pool.imap(
                    prosody_segment_worker,
                    [(segment, str(audio_path), sample_rate, max_samples) for segment in segments],
                ):
                    if error:
                        errors += 1
                        progress.write(f"    [Prosody] Warning: entry {idx} failed: {error}")
                    else:
                        if volume:
                            transcript.entries[idx].volume = volume
                        if speech_rate:
                            transcript.entries[idx].speech_rate = speech_rate
                        if pitch_trend:
                            transcript.entries[idx].pitch_trend = pitch_trend
                    progress.update(1)
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

    def _device_string(self) -> str:
        from yorishiro.utils import get_device

        return get_device()

    def _clear_torch_cache(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
