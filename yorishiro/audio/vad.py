"""Voice activity detection runtime for `film.audio.vad`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


_SILERO_SAMPLING_RATE = 16000
_SILERO_WINDOW_SAMPLES = 512


@dataclass(frozen=True)
class VadConfig:
    vad_backend: str = "silero-vad"
    vad_profile: str = "balanced"


@dataclass(frozen=True)
class _VadProfile:
    name: str
    threshold: float
    neg_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    max_speech_duration_s: float
    max_boundary_shift_ms: int
    rescue_guard_gap_seconds: float


_VAD_PROFILES: dict[str, _VadProfile] = {
    "balanced": _VadProfile(
        name="balanced",
        threshold=0.40,
        neg_threshold=0.22,
        min_speech_duration_ms=140,
        min_silence_duration_ms=220,
        speech_pad_ms=120,
        max_speech_duration_s=30.0,
        max_boundary_shift_ms=96,
        rescue_guard_gap_seconds=0.20,
    ),
    "recall": _VadProfile(
        name="recall",
        threshold=0.30,
        neg_threshold=0.15,
        min_speech_duration_ms=100,
        min_silence_duration_ms=150,
        speech_pad_ms=180,
        max_speech_duration_s=30.0,
        max_boundary_shift_ms=120,
        rescue_guard_gap_seconds=0.25,
    ),
    "precise": _VadProfile(
        name="precise",
        threshold=0.50,
        neg_threshold=0.35,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=80,
        max_speech_duration_s=30.0,
        max_boundary_shift_ms=64,
        rescue_guard_gap_seconds=0.18,
    ),
}


class VadRunner:
    """Run voice activity detection and persist `vad.json`."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()

    def release_models(self) -> None:
        pass

    def run(
        self,
        audio_path: Path,
        output_dir: Path,
        nonvoice_path: Path | None = None,
    ) -> list[dict[str, float]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [VAD] Running on {audio_path.name} ...")
        segments, debug_info = self._run_vad(audio_path, nonvoice_path=nonvoice_path)
        (output_dir / "vad.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "vad.debug.json").write_text(
            json.dumps(debug_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [VAD] Done — {len(segments)} speech segment(s)")
        return segments

    def _run_vad(
        self,
        audio_path: Path,
        *,
        nonvoice_path: Path | None = None,
    ) -> tuple[list[dict[str, float]], dict[str, Any]]:
        from silero_vad import get_speech_timestamps
        from silero_vad import load_silero_vad
        from silero_vad import read_audio

        profile = self._profile()
        model = load_silero_vad()
        wav = read_audio(str(audio_path), sampling_rate=_SILERO_SAMPLING_RATE)
        audio_duration = float(sf.info(str(audio_path)).duration)

        raw_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=_SILERO_SAMPLING_RATE,
            return_seconds=False,
            threshold=profile.threshold,
            neg_threshold=profile.neg_threshold,
            min_speech_duration_ms=profile.min_speech_duration_ms,
            min_silence_duration_ms=profile.min_silence_duration_ms,
            speech_pad_ms=profile.speech_pad_ms,
            max_speech_duration_s=profile.max_speech_duration_s,
        )
        base_segments = [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "source": "voice",
            }
            for item in raw_timestamps
            if int(item["end"]) > int(item["start"])
        ]

        primary_quality = self._estimate_segment_quality(
            wav,
            model,
            base_segments,
            progress_desc="    [VAD] Quality (voice)",
        )
        primary_segments = [
            {
                "start": item["start"],
                "end": item["end"],
                "quality": primary_quality[idx],
                "source": "voice",
            }
            for idx, item in enumerate(base_segments)
        ]

        adjusted_segments, adjustments = self._adjust_boundaries(
            primary_segments,
            wav,
            profile=profile,
        )
        post_segments, dropped_segments = self._post_process_segments(
            adjusted_segments,
            audio_duration=audio_duration,
        )

        rescue_added: list[dict[str, float | str]] = []
        if nonvoice_path and nonvoice_path.exists():
            nonvoice_wav = read_audio(
                str(nonvoice_path), sampling_rate=_SILERO_SAMPLING_RATE
            )
            mixed_wav = self._mix_audio(wav, nonvoice_wav)
            rescue_candidates = self._detect_rescue_segments(mixed_wav, model)
            rescue_quality = self._estimate_segment_quality(
                mixed_wav,
                model,
                rescue_candidates,
                progress_desc="    [VAD] Quality (rescue)",
            )
            rescue_scored = [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "quality": rescue_quality[idx],
                    "source": "mixed-rescue",
                }
                for idx, item in enumerate(rescue_candidates)
            ]
            post_segments, rescue_added = self._add_rescue_segments(
                post_segments,
                rescue_scored,
                guard_gap_seconds=profile.rescue_guard_gap_seconds,
                audio_duration=audio_duration,
            )

        if not post_segments:
            post_segments = [
                {
                    "start": 0.0,
                    "end": round(audio_duration, 3),
                    "quality": 0.0,
                }
            ]

        final_segments = [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "quality": float(item["quality"]),
            }
            for item in post_segments
        ]

        total_speech = sum(
            max(0.0, seg["end"] - seg["start"]) for seg in final_segments
        )
        mean_quality = (
            sum(seg["quality"] for seg in final_segments) / len(final_segments)
            if final_segments
            else 0.0
        )
        max_shift_ms = max(
            (abs(float(item["shift_ms"])) for item in adjustments), default=0.0
        )
        debug_info: dict[str, Any] = {
            "audio_duration": round(audio_duration, 3),
            "profile": profile.name,
            "primary_segments": [
                {
                    "start": self._seconds(int(item["start"])),
                    "end": self._seconds(int(item["end"])),
                    "quality": float(item["quality"]),
                }
                for item in primary_segments
            ],
            "rescue_segments_added": [
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "quality": float(item["quality"]),
                }
                for item in rescue_added
            ],
            "dropped_segments": dropped_segments,
            "adjustments": adjustments,
            "stats": {
                "primary_count": len(primary_segments),
                "rescue_added_count": len(rescue_added),
                "dropped_count": len(dropped_segments),
                "final_count": len(final_segments),
                "total_speech_seconds": round(total_speech, 3),
                "mean_quality": round(mean_quality, 4),
                "low_quality_count": sum(
                    1 for seg in final_segments if seg["quality"] < 0.3
                ),
                "largest_boundary_shift_ms": round(max_shift_ms, 2),
            },
        }
        return final_segments, debug_info

    def _profile(self) -> _VadProfile:
        key = (self.config.vad_profile or "balanced").strip().lower()
        return _VAD_PROFILES.get(key, _VAD_PROFILES["balanced"])

    def _estimate_segment_quality(
        self,
        wav: Any,
        model: Any,
        segments: list[dict[str, int | str]],
        *,
        progress_desc: str | None = None,
    ) -> list[float]:
        if not segments:
            return []
        probs = self._speech_probabilities(
            wav,
            model,
            progress_desc=progress_desc,
        )
        if not probs:
            return [0.0 for _ in segments]

        qualities: list[float] = []
        for item in segments:
            start = int(item["start"])
            end = int(item["end"])
            if end <= start:
                qualities.append(0.0)
                continue
            start_idx = max(0, start // _SILERO_WINDOW_SAMPLES)
            end_idx = max(
                start_idx + 1,
                (end + _SILERO_WINDOW_SAMPLES - 1) // _SILERO_WINDOW_SAMPLES,
            )
            window_probs = probs[start_idx : min(len(probs), end_idx)]
            if not window_probs:
                qualities.append(0.0)
                continue
            mean_prob = sum(window_probs) / len(window_probs)
            qualities.append(round(float(mean_prob), 4))
        return qualities

    def _speech_probabilities(
        self,
        wav: Any,
        model: Any,
        *,
        progress_desc: str | None = None,
    ) -> list[float]:
        import torch
        from tqdm import tqdm

        if hasattr(model, "reset_states"):
            model.reset_states()
        audio_len = int(wav.shape[0])
        iter_range = range(0, audio_len, _SILERO_WINDOW_SAMPLES)
        if progress_desc and audio_len > 0:
            total_windows = (
                audio_len + _SILERO_WINDOW_SAMPLES - 1
            ) // _SILERO_WINDOW_SAMPLES
            starts = tqdm(
                iter_range,
                total=total_windows,
                desc=progress_desc,
                unit="win",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        else:
            starts = iter_range
        probs: list[float] = []
        with torch.no_grad():
            for start in starts:
                chunk = wav[start : start + _SILERO_WINDOW_SAMPLES]
                if int(chunk.shape[0]) < _SILERO_WINDOW_SAMPLES:
                    pad = _SILERO_WINDOW_SAMPLES - int(chunk.shape[0])
                    chunk = torch.nn.functional.pad(
                        chunk, (0, pad), "constant", value=0.0
                    )
                probs.append(float(model(chunk, _SILERO_SAMPLING_RATE).item()))
        return probs

    def _adjust_boundaries(
        self,
        segments: list[dict[str, float | int | str]],
        wav: Any,
        *,
        profile: _VadProfile,
    ) -> tuple[list[dict[str, float]], list[dict[str, float | int | str]]]:
        adjusted: list[dict[str, float]] = []
        details: list[dict[str, float | int | str]] = []
        max_shift = int((_SILERO_SAMPLING_RATE * profile.max_boundary_shift_ms) / 1000)
        peak = float(wav.abs().max().item()) if int(wav.shape[0]) else 0.0
        amplitude_floor = max(0.004, peak * 0.08)

        for idx, item in enumerate(segments):
            raw_start = int(item["start"])
            raw_end = int(item["end"])
            quality = float(item["quality"])
            if raw_end <= raw_start:
                continue
            start = self._snap_to_energy(
                wav,
                raw_start,
                direction="left",
                max_shift=max_shift,
                amplitude_floor=amplitude_floor,
            )
            end = self._snap_to_energy(
                wav,
                raw_end,
                direction="right",
                max_shift=max_shift,
                amplitude_floor=amplitude_floor,
            )
            if end <= start:
                end = raw_end
                start = raw_start

            adjusted.append(
                {
                    "start": self._seconds(start),
                    "end": self._seconds(end),
                    "quality": quality,
                }
            )
            details.append(
                {
                    "index": idx,
                    "raw_start": self._seconds(raw_start),
                    "raw_end": self._seconds(raw_end),
                    "adjusted_start": self._seconds(start),
                    "adjusted_end": self._seconds(end),
                    "shift_ms": round(
                        ((start - raw_start) / _SILERO_SAMPLING_RATE) * 1000.0, 2
                    ),
                }
            )
        return adjusted, details

    def _snap_to_energy(
        self,
        wav: Any,
        point: int,
        *,
        direction: str,
        max_shift: int,
        amplitude_floor: float,
    ) -> int:
        import torch

        total = int(wav.shape[0])
        step = 160
        left = max(0, point - max_shift)
        right = min(total, point + max_shift)
        best = point
        if direction == "left":
            iterator = range(point, left - 1, -step)
        else:
            iterator = range(point, right + 1, step)

        for center in iterator:
            win_start = max(0, center - step)
            win_end = min(total, center + step)
            if win_end <= win_start:
                continue
            window = wav[win_start:win_end]
            energy = float(torch.mean(torch.abs(window)).item())
            if energy >= amplitude_floor:
                best = center
                break
        return int(max(0, min(total, best)))

    def _post_process_segments(
        self,
        segments: list[dict[str, float]],
        *,
        audio_duration: float,
    ) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
        ordered = sorted(segments, key=lambda item: (item["start"], item["end"]))
        dropped: list[dict[str, float | str]] = []
        clamped: list[dict[str, float]] = []
        min_duration = 0.04
        for item in ordered:
            start = max(0.0, float(item["start"]))
            end = min(audio_duration, float(item["end"]))
            if end - start < min_duration:
                dropped.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "quality": float(item["quality"]),
                        "reason": "too_short",
                    }
                )
                continue
            clamped.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "quality": float(item["quality"]),
                }
            )

        repaired: list[dict[str, float]] = []
        for item in clamped:
            if not repaired:
                repaired.append(item)
                continue
            prev = repaired[-1]
            if item["start"] >= prev["end"]:
                repaired.append(item)
                continue

            midpoint = round((prev["end"] + item["start"]) / 2.0, 3)
            prev_min_end = round(prev["start"] + min_duration, 3)
            next_max_start = round(item["end"] - min_duration, 3)
            cut = min(max(midpoint, prev_min_end), next_max_start)
            prev["end"] = cut
            item["start"] = cut
            if prev["end"] - prev["start"] < min_duration:
                dropped.append(
                    {
                        "start": prev["start"],
                        "end": prev["end"],
                        "quality": prev["quality"],
                        "reason": "overlap_repair_prev_too_short",
                    }
                )
                repaired.pop()
            if item["end"] - item["start"] < min_duration:
                dropped.append(
                    {
                        "start": item["start"],
                        "end": item["end"],
                        "quality": item["quality"],
                        "reason": "overlap_repair_next_too_short",
                    }
                )
                continue
            repaired.append(item)

        return repaired, dropped

    def _detect_rescue_segments(
        self,
        mixed_wav: Any,
        model: Any,
    ) -> list[dict[str, int | str]]:
        from silero_vad import get_speech_timestamps

        profile = _VAD_PROFILES["recall"]
        timestamps = get_speech_timestamps(
            mixed_wav,
            model,
            sampling_rate=_SILERO_SAMPLING_RATE,
            return_seconds=False,
            threshold=profile.threshold,
            neg_threshold=profile.neg_threshold,
            min_speech_duration_ms=profile.min_speech_duration_ms,
            min_silence_duration_ms=profile.min_silence_duration_ms,
            speech_pad_ms=profile.speech_pad_ms,
            max_speech_duration_s=profile.max_speech_duration_s,
        )
        return [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "source": "mixed-rescue",
            }
            for item in timestamps
            if int(item["end"]) > int(item["start"])
        ]

    def _add_rescue_segments(
        self,
        base_segments: list[dict[str, float]],
        rescue_segments: list[dict[str, float | str]],
        *,
        guard_gap_seconds: float,
        audio_duration: float,
    ) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
        accepted = list(base_segments)
        added: list[dict[str, float | str]] = []

        for rescue in sorted(
            rescue_segments, key=lambda item: (float(item["start"]), float(item["end"]))
        ):
            start = max(0.0, float(rescue["start"]))
            end = min(audio_duration, float(rescue["end"]))
            if end - start < 0.08:
                continue
            near_existing = any(
                (start < seg["end"] + guard_gap_seconds)
                and (end > seg["start"] - guard_gap_seconds)
                for seg in accepted
            )
            if near_existing:
                continue
            item = {
                "start": round(start, 3),
                "end": round(end, 3),
                "quality": float(rescue["quality"]),
            }
            accepted.append(item)
            added.append({**item, "source": str(rescue.get("source", "mixed-rescue"))})

        accepted.sort(key=lambda item: (item["start"], item["end"]))
        accepted, _ = self._post_process_segments(
            accepted, audio_duration=audio_duration
        )
        return accepted, added

    def _mix_audio(self, voice_wav: Any, nonvoice_wav: Any) -> Any:
        import torch

        voice_len = int(voice_wav.shape[0])
        nonvoice_len = int(nonvoice_wav.shape[0])
        max_len = max(voice_len, nonvoice_len)
        if voice_len < max_len:
            voice_wav = torch.nn.functional.pad(
                voice_wav, (0, max_len - voice_len), "constant", value=0.0
            )
        if nonvoice_len < max_len:
            nonvoice_wav = torch.nn.functional.pad(
                nonvoice_wav, (0, max_len - nonvoice_len), "constant", value=0.0
            )
        mixed = voice_wav + nonvoice_wav
        return mixed.clamp(min=-1.0, max=1.0)

    def _seconds(self, sample_idx: int) -> float:
        return round(float(sample_idx) / _SILERO_SAMPLING_RATE, 3)
