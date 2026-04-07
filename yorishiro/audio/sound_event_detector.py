"""Sound event detection using CLAP.

Detects non-speech sounds: ambient, sound effects, non-speech vocals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import librosa
import numpy as np
import torch
from dataclasses import dataclass
from tqdm import tqdm

from yorishiro.models.film_models import SoundEvent
from yorishiro.utils import get_device


@dataclass
class SoundEventDetectorConfig:
    backend: str = "clap"
    model: str = "laion/larger_clap_general"
    vocal_similarity_threshold: float = 0.22
    ambience_similarity_threshold: float = 0.20
    vocal_margin_threshold: float = 0.03
    ambience_margin_threshold: float = 0.025
    silence_margin_threshold: float = 0.03
    voice_min_rms_db: float = -38.0
    ambience_min_rms_db: float = -35.0
    speech_padding_seconds: float = 0.20


VOICE_EVENT_PROMPTS = [
    "crying",
    "laughing",
    "joyful exclamation",
    "gasp or alarm",
    "scream or shout",
    "heavy breathing",
    "nonverbal vocalization",
    "silence",
]

NONVOICE_EVENT_PROMPTS = [
    "rain",
    "wind",
    "water",
    "crowd noise",
    "instrumental music",
    "applause",
    "vehicle or engine",
    "quiet room tone",
    "impact or noise",
    "silence",
]


class SoundEventDetector:
    """Detects non-speech sound events using CLAP."""

    def __init__(self, config: SoundEventDetectorConfig | None = None):
        self.config = config or SoundEventDetectorConfig()
        self._clap_model = None
        self._clap_processor = None
        self._text_features_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @staticmethod
    def _embedding_tensor(output: torch.Tensor | object) -> torch.Tensor:
        """Extract the embedding tensor from CLAP helper outputs across transformers versions."""
        if isinstance(output, torch.Tensor):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        if isinstance(output, tuple) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        raise TypeError(f"Unsupported CLAP output type: {type(output)!r}")

    @staticmethod
    def _merge_adjacent_events(events: list[SoundEvent], gap_tolerance: float = 0.6) -> list[SoundEvent]:
        if not events:
            return []

        merged = [events[0]]
        for event in events[1:]:
            prev = merged[-1]
            if (
                event.description == prev.description
                and event.event_type == prev.event_type
                and event.start <= prev.end + gap_tolerance
            ):
                prev.end = max(prev.end, event.end)
                continue
            merged.append(event)
        return merged

    @staticmethod
    def _window_rms_db(chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
        return 20.0 * math.log10(rms + 1e-9)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _interval_overlap_fraction(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
        duration = max(end - start, 1e-6)
        overlap = 0.0
        for i_start, i_end in intervals:
            if i_end <= start:
                continue
            if i_start >= end:
                break
            overlap += max(0.0, min(end, i_end) - max(start, i_start))
        return overlap / duration

    def _speech_intervals(self, transcript_path: Path) -> list[tuple[float, float]]:
        if not transcript_path.exists():
            return []
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
            padding = self.config.speech_padding_seconds
            intervals = [
                (max(0.0, float(entry.get("start", 0.0)) - padding), float(entry.get("end", 0.0)) + padding)
                for entry in entries
            ]
            return self._merge_intervals(intervals)
        except Exception:
            return []

    def _filter_candidates(self, candidates: list[dict], *, require_support: bool) -> list[SoundEvent]:
        if not candidates:
            return []

        filtered: list[dict] = []
        n = len(candidates)
        for i, cand in enumerate(candidates):
            prev_same = i > 0 and candidates[i - 1]["description"] == cand["description"] and candidates[i - 1]["end"] >= cand["start"] - 0.6
            next_same = i + 1 < n and candidates[i + 1]["description"] == cand["description"] and candidates[i + 1]["start"] <= cand["end"] + 0.6
            if not require_support or prev_same or next_same:
                filtered.append(cand)

        events = [
            SoundEvent(
                start=float(c["start"]),
                end=float(c["end"]),
                event_type=c["event_type"],
                description=c["description"],
            )
            for c in filtered
        ]
        return self._merge_adjacent_events(events)

    def detect(
        self,
        voice_path: Path,
        nonvoice_path: Path,
        output_dir: Path,
        transcript_path: Path | None = None,
        force: bool = False,
    ) -> list[SoundEvent]:
        """Detect sound events in voice and non-voice stems.

        Returns list of SoundEvent.
        Caches to output_dir / sound_events.json.
        """
        cache_file = output_dir / "sound_events.json"

        if not force and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return [SoundEvent(**e) for e in cached.get("events", [])]
            except Exception:
                pass

        speech_intervals = self._speech_intervals(transcript_path) if transcript_path is not None else []

        print(f"  [SoundEventDetector] Processing {voice_path.name} for vocal events ...")
        voice_events = self._detect_vocal_events(
            voice_path,
            speech_intervals,
        )
        print(f"  [SoundEventDetector] Processing {nonvoice_path.name} for ambience ...")
        nonvoice_events = self._detect_ambience_events(
            nonvoice_path,
        )
        events = sorted(voice_events + nonvoice_events, key=lambda e: (e.start, e.end, e.description))
        events = self._merge_adjacent_events(events)

        result = {"events": [e.model_dump() for e in events]}
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  [SoundEventDetector] Detected {len(events)} sound events")
        return events

    def _text_features(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        key = tuple(prompts)
        cached = self._text_features_cache.get(key)
        if cached is not None:
            return cached

        assert self._clap_processor is not None
        assert self._clap_model is not None
        text_inputs = self._clap_processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self._embedding_tensor(self._clap_model.get_text_features(**text_inputs))
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        self._text_features_cache[key] = text_features
        return text_features

    def _classify_window(
        self,
        chunk: np.ndarray,
        prompts: list[str],
        text_features: torch.Tensor,
        *,
        similarity_threshold: float,
        margin_threshold: float,
    ) -> str | None:
        assert self._clap_model is not None
        assert self._clap_processor is not None
        device = next(self._clap_model.parameters()).device

        audio_inputs = self._clap_processor(
            audio=chunk,
            return_tensors="pt",
            sampling_rate=48000,
        )
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            audio_features = self._embedding_tensor(self._clap_model.get_audio_features(**audio_inputs))
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
            sim_row = (audio_features @ text_features.T)[0]
            top_k = min(2, sim_row.shape[0])
            top_sims, top_indices = torch.topk(sim_row, k=top_k)

        top_sim = float(top_sims[0])
        top_idx = int(top_indices[0])
        second_sim = float(top_sims[1]) if top_k > 1 else -1.0e9
        if top_sim < similarity_threshold:
            return None
        if top_sim - second_sim < margin_threshold:
            return None

        label = prompts[top_idx]
        if label == "silence":
            return None

        if "silence" in prompts:
            silence_idx = prompts.index("silence")
            silence_sim = float(sim_row[silence_idx])
            if silence_sim + self.config.silence_margin_threshold >= top_sim:
                return None

        return label

    @staticmethod
    def _normalize_label(label: str) -> str:
        mapping = {
            "crying": "crying_or_sobbing",
            "joyful exclamation": "joyful_exclamation",
            "gasp or alarm": "gasp_or_alarm",
            "scream or shout": "scream_or_shout",
            "heavy breathing": "heavy_breathing",
            "nonverbal vocalization": "nonverbal_vocalization",
            "water": "water",
            "crowd noise": "crowd",
            "instrumental music": "music",
            "vehicle or engine": "vehicle_or_engine",
            "quiet room tone": "room_tone",
            "impact or noise": "impact_or_noise",
        }
        return mapping.get(label, label)

    def _detect_vocal_events(
        self,
        audio_path: Path,
        speech_intervals: list[tuple[float, float]],
    ) -> list[SoundEvent]:
        try:
            from transformers import ClapModel, ClapProcessor

            if self._clap_model is None:
                device = torch.device(get_device())
                self._clap_model = ClapModel.from_pretrained(self.config.model)
                self._clap_model = self._clap_model.to(device)  # type: ignore
                self._clap_processor = ClapProcessor.from_pretrained(self.config.model)

            device = next(self._clap_model.parameters()).device
            text_features = self._text_features(VOICE_EVENT_PROMPTS, device)

            audio, sr = librosa.load(str(audio_path), sr=48000)
            duration = len(audio) / sr

            window_size = 2.5
            hop_size = 1.25
            candidates: list[dict] = []

            max_start = max(0.0, duration - window_size)
            starts: list[float] = []
            cur = 0.0
            while cur <= max_start + 1e-6:
                starts.append(cur)
                cur += hop_size
            if not starts:
                starts = [0.0]

            print(f"  [SoundEventDetector] Scanning {len(starts)} voice window(s) ...")
            for start in tqdm(starts, desc="  [SoundEventDetector] Voice", unit="window"):
                end = min(start + window_size, duration)
                chunk = audio[int(start * sr):int(end * sr)]
                if len(chunk) == 0:
                    continue
                if self._window_rms_db(chunk) < self.config.voice_min_rms_db:
                    continue
                if self._interval_overlap_fraction(start, end, speech_intervals) > 0.15:
                    continue

                label = self._classify_window(
                    chunk,
                    VOICE_EVENT_PROMPTS,
                    text_features,
                    similarity_threshold=self.config.vocal_similarity_threshold,
                    margin_threshold=self.config.vocal_margin_threshold,
                )
                if label is None:
                    continue

                candidates.append({
                    "start": float(start),
                    "end": float(end),
                    "event_type": "non_speech_vocal",
                    "description": self._normalize_label(label),
                })

            return self._filter_candidates(candidates, require_support=False)

        except Exception as e:
            print(f"  [SoundEventDetector] Error: {e}, returning empty events")
            return []

    def _detect_ambience_events(self, audio_path: Path) -> list[SoundEvent]:
        try:
            from transformers import ClapModel, ClapProcessor

            if self._clap_model is None:
                device = torch.device(get_device())
                self._clap_model = ClapModel.from_pretrained(self.config.model)
                self._clap_model = self._clap_model.to(device)  # type: ignore
                self._clap_processor = ClapProcessor.from_pretrained(self.config.model)

            device = next(self._clap_model.parameters()).device
            text_features = self._text_features(NONVOICE_EVENT_PROMPTS, device)

            audio, sr = librosa.load(str(audio_path), sr=48000)
            duration = len(audio) / sr
            window_size = 4.0
            hop_size = 2.0
            candidates: list[dict] = []

            max_start = max(0.0, duration - window_size)
            starts: list[float] = []
            cur = 0.0
            while cur <= max_start + 1e-6:
                starts.append(cur)
                cur += hop_size
            if not starts:
                starts = [0.0]

            print(f"  [SoundEventDetector] Scanning {len(starts)} nonvoice window(s) ...")
            for start in tqdm(starts, desc="  [SoundEventDetector] Nonvoice", unit="window"):
                end = min(start + window_size, duration)
                chunk = audio[int(start * sr):int(end * sr)]
                if len(chunk) == 0:
                    continue
                if self._window_rms_db(chunk) < self.config.ambience_min_rms_db:
                    continue

                flatness = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
                if flatness < 0.01:
                    continue

                label = self._classify_window(
                    chunk,
                    NONVOICE_EVENT_PROMPTS,
                    text_features,
                    similarity_threshold=self.config.ambience_similarity_threshold,
                    margin_threshold=self.config.ambience_margin_threshold,
                )
                if label is None:
                    continue

                candidates.append({
                    "start": float(start),
                    "end": float(end),
                    "event_type": "ambient",
                    "description": self._normalize_label(label),
                })

            return self._filter_candidates(candidates, require_support=True)

        except Exception as e:
            print(f"  [SoundEventDetector] Error: {e}, returning empty events")
            return []
