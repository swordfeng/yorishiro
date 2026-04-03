"""Sound event detection using CLAP.

Detects non-speech sounds: ambient, sound effects, non-speech vocals.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import torch
from dataclasses import dataclass

from yorishiro.models.film_models import SoundEvent
from yorishiro.utils import get_device


@dataclass
class SoundEventDetectorConfig:
    backend: str = "clap"
    model: str = "laion/larger_clap_general"
    threshold: float = 0.3


SOUND_PROMPTS = [
    "crying",
    "laughing",
    "breathing",
    "gasping",
    "sighing",
    "heartbeat",
    "moaning",
    "screaming",
    "footsteps",
    "door opening",
    "door closing",
    "glass breaking",
    "explosion",
    "gunshot",
    "car engine",
    "rain",
    "wind",
    "water flowing",
    "thunder",
    "clock ticking",
    "phone ringing",
    "bell",
    "siren",
    "crowd",
    "applause",
    "silence",
]


class SoundEventDetector:
    """Detects non-speech sound events using CLAP."""

    def __init__(self, config: SoundEventDetectorConfig | None = None):
        self.config = config or SoundEventDetectorConfig()
        self._clap_model = None
        self._clap_processor = None

    def detect(
        self,
        audio_path: Path,
        output_dir: Path,
        transcript_end: float = 0.0,
        force: bool = False,
    ) -> list[SoundEvent]:
        """Detect sound events in audio.

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

        print(f"  [SoundEventDetector] Processing {audio_path.name} ...")
        events = self._detect_events(audio_path, transcript_end)

        result = {"events": [e.model_dump() for e in events]}
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  [SoundEventDetector] Detected {len(events)} sound events")
        return events

    def _detect_events(self, audio_path: Path, transcript_end: float) -> list[SoundEvent]:
        """Run CLAP-based sound event detection."""
        try:
            from transformers import ClapModel, ClapProcessor

            if self._clap_model is None:
                device = torch.device(get_device())
                self._clap_model = ClapModel.from_pretrained(self.config.model)
                self._clap_model = self._clap_model.to(device)  # type: ignore
                self._clap_processor = ClapProcessor.from_pretrained(self.config.model)

            device = next(self._clap_model.parameters()).device

            audio, sr = librosa.load(str(audio_path), sr=48000)
            duration = len(audio) / sr

            window_size = 5.0
            hop_size = 2.5
            events = []

            for start in range(0, int(duration - window_size), int(hop_size)):
                end = start + window_size

                chunk = audio[int(start * sr):int(end * sr)]

                assert self._clap_processor is not None
                inputs = self._clap_processor(
                    audio=chunk,
                    text=SOUND_PROMPTS,
                    return_tensors="pt",
                    sampling_rate=48000,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clap_model(**inputs)
                    probs = torch.softmax(outputs.logits_per_audio, dim=-1)
                    max_prob, max_idx = torch.max(probs[0], dim=-1)

                if max_prob > self.config.threshold:
                    event_type = self._classify_event_type(SOUND_PROMPTS[max_idx])
                    events.append(SoundEvent(
                        start=float(start),
                        end=float(end),
                        event_type=event_type,
                        description=SOUND_PROMPTS[max_idx],
                    ))

            return events

        except Exception as e:
            print(f"  [SoundEventDetector] Error: {e}, returning empty events")
            return []

    def _classify_event_type(self, prompt: str) -> str:
        """Classify event type based on prompt."""
        non_speech_vocals = {"crying", "laughing", "breathing", "gasping", "sighing", "heartbeat", "moaning", "screaming"}
        ambient = {"rain", "wind", "thunder", "water flowing", "car engine"}
        sfx = {"footsteps", "door opening", "door closing", "glass breaking", "explosion", "gunshot", "clock ticking", "phone ringing", "bell", "siren"}

        if prompt in non_speech_vocals:
            return "non_speech_vocal"
        elif prompt in ambient:
            return "ambient"
        elif prompt in sfx:
            return "sfx"
        elif prompt == "silence":
            return "silence"
        else:
            return "ambient"